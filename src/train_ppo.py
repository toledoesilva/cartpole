"""Training script for the PPO agent on CartPole-v1.

This file intentionally keeps runtime under a main guard so importing the
module doesn't start training (helps interactive debugging and `--help`).
"""

import argparse
from typing import Any, Tuple

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from src.env_utils import CustomCartPoleEnv
from src.ppo_agent import PPOAgent
import json
import time


def _unwrap_reset(reset_ret: Any):
    """Normalize env.reset() return to observation only.

    Supports both Gym (obs) and Gymnasium (obs, info) signatures.
    """
    if isinstance(reset_ret, tuple) or isinstance(reset_ret, list):
        return reset_ret[0]
    return reset_ret


def _unwrap_step(step_ret: Tuple):
    """Normalize env.step() return to (next_state, reward, done, info).

    Supports both Gym (4-tuple) and Gymnasium (5-tuple).
    """
    if len(step_ret) == 5:
        next_state, reward, terminated, truncated, info = step_ret
        done = bool(terminated or truncated)
        return next_state, reward, done, info
    if len(step_ret) == 4:
        next_state, reward, done, info = step_ret
        return next_state, reward, bool(done), info
    raise ValueError(f"Unexpected step return signature: {step_ret}")


def save_state(state_path: str, episodes_completed: int, best_avg: float) -> None:
    """Persist minimal training state to disk.

    This is separated from `main` so we keep `main` small and easier to lint.
    """
    state = {
        "episodes_completed": int(episodes_completed),
        "best_avg_reward": float(best_avg),
        "last_saved": time.time(),
    }
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        # non-critical: don't crash training because state couldn't be written
        pass


def _run_training_loop(
    env: CustomCartPoleEnv,
    agent: PPOAgent,
    start_episode: int,
    episodes: int,
    rollout_length: int,
    save_path: str,
    latest_interval: int,
    best_avg_reward: float,
    state_path: str,
):
    """Run the main training loop. Returns collected rewards and final best_avg_reward."""
    all_rewards: list[float] = []

    for episode in range(start_episode, episodes):
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        reset_ret = env.reset()
        state = _unwrap_reset(reset_ret)

        # Debugging: Print the state to verify its structure
        print(f"Initial state: {state}")

        # Ensure state is non-empty numeric
        if not isinstance(state, (list, np.ndarray)) or len(state) == 0:
            raise ValueError(f"Unexpected state format: {state}")

        episode_rewards = []
        per_env_return = 0.0
        env_episodes_this_iter = 0

        for _ in range(rollout_length):
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(
                0
            )
            logits, value = agent.model(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

            step_ret = env.step(action)
            next_state, reward, done, _info = _unwrap_step(step_ret)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(dist.log_prob(torch.tensor(action)).item())
            values.append(value.item())

            per_env_return += float(reward)
            state = next_state
            episode_rewards.append(reward)

            if done:
                env_episodes_this_iter += 1
                all_rewards.append(per_env_return)
                print(f"Env episode finished — reward={per_env_return}")
                per_env_return = 0.0
                reset_ret = env.reset()
                state = _unwrap_reset(reset_ret)

        last_value = (
            0
            if done
            else agent.model(torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0))[1].item()
        )
        values.append(last_value)
        advantages = agent.compute_advantages(np.array(rewards), np.array(values), np.array(dones))
        returns = advantages + np.array(values[:-1])

        trajectories = {
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "returns": returns,
            "advantages": advantages,
        }
        agent.update(trajectories)

        if len(all_rewards) > 0:
            avg_reward = float(np.mean(all_rewards[-100:]))
        else:
            avg_reward = 0.0

        print(f"Iter {episode + 1}/{episodes} | env_eps: {env_episodes_this_iter} | Avg100: {avg_reward}")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(agent.model.state_dict(), save_path)

        try:
            if latest_interval > 0 and ((episode + 1) % latest_interval) == 0:
                latest_path = os.path.join(os.path.dirname(save_path) or ".", "latest_model.pt")
                os.makedirs(os.path.dirname(latest_path), exist_ok=True)
                torch.save(agent.model.state_dict(), latest_path)
        except Exception:
            pass

        save_state(state_path, episode + 1, best_avg_reward)

    return all_rewards, best_avg_reward


def main(
    episodes: int = 1000,
    rollout_length: int = 2048,
    save_path: str = "models/best_model.pt",
    latest_interval: int = 10,
):
    # Create environment
    env = CustomCartPoleEnv(render_mode="rgb_array")
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize PPO agent
    agent = PPOAgent(input_dim, action_dim)

    # Training loop
    best_avg_reward = -float("inf")
    all_rewards = []
    state_path = "training_state.json"

    # attempt to resume if a previous state exists; actual resume control is via CLI
    episodes_completed = 0
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            episodes_completed = int(prev.get("episodes_completed", 0))
            # Load best_avg_reward but sanity-check it. If the value looks corrupted
            # (extremely large) or non-positive, ignore it so it doesn't block saves.
            loaded_best = float(prev.get("best_avg_reward", best_avg_reward))
            if loaded_best > 1e6 or loaded_best <= 0.0 or not np.isfinite(loaded_best):
                print(f"Warning: loaded best_avg_reward={loaded_best} invalid; resetting to -inf.")
                best_avg_reward = -float("inf")
            else:
                best_avg_reward = loaded_best
        except Exception:
            episodes_completed = 0
    # Loop from episodes_completed to target `episodes` so we can resume

    # Loop from episodes_completed to target `episodes` so we can resume
    start_episode = int(episodes_completed)
    if start_episode >= episodes:
        print(
            f"Training already completed ({start_episode} >= {episodes}). Nothing to do."
        )
        return

    # Delegate the heavy lifting to a helper to keep main() concise and testable
    all_rewards, best_avg_reward = _run_training_loop(
        env=env,
        agent=agent,
        start_episode=start_episode,
        episodes=episodes,
        rollout_length=rollout_length,
        save_path=save_path,
        latest_interval=latest_interval,
        best_avg_reward=best_avg_reward,
        state_path=state_path,
    )

    # Plot training rewards
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance")
    plt.show()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on CartPole-v1")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--rollout-length", type=int, default=2048, help="Rollout length per episode"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/best_model.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--latest-interval",
        type=int,
        default=10,
        help="Save a rolling latest_model.pt every N episodes (0 to disable)",
    )
    args = parser.parse_args()

    main(
        episodes=args.episodes,
        rollout_length=args.rollout_length,
        save_path=args.save_path,
        latest_interval=args.latest_interval,
    )
