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
import random


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


def _save_checkpoint(
    checkpoint_path: str, agent: PPOAgent, episodes_completed: int, best_avg: float
) -> None:
    """Save a full checkpoint including model, optimizers and RNG states.

    This allows exact resumption of training dynamics (optimizer states and RNGs).
    The function is best-effort and will not raise on failure.
    """
    try:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        ckpt = {
            "model_state_dict": agent.model.state_dict(),
            "policy_optimizer": agent.policy_optimizer.state_dict(),
            "value_optimizer": agent.value_optimizer.state_dict(),
            "episodes_completed": int(episodes_completed),
            "best_avg_reward": float(best_avg),
            "py_random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }
        # Save to disk using torch so tensors serialize properly
        torch.save(ckpt, checkpoint_path)
    except Exception:
        # non-critical
        pass


def _load_checkpoint(
    checkpoint_path: str, agent: PPOAgent
) -> dict | None:  # noqa: C901
    """Attempt to load a full checkpoint and restore model+optimizers+RNGs.

    Returns the loaded metadata dict on success, or None on failure.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location=agent.device)
        if "model_state_dict" in ckpt:
            agent.model.load_state_dict(ckpt["model_state_dict"])
        # restore optimizers if keys present
        if "policy_optimizer" in ckpt:
            try:
                agent.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
            except Exception:
                pass
        if "value_optimizer" in ckpt:
            try:
                agent.value_optimizer.load_state_dict(ckpt["value_optimizer"])
            except Exception:
                pass
        # restore RNGs
        try:
            if "py_random_state" in ckpt:
                random.setstate(ckpt["py_random_state"])
            if "np_random_state" in ckpt:
                np.random.set_state(ckpt["np_random_state"])
            if "torch_rng_state" in ckpt:
                torch.set_rng_state(ckpt["torch_rng_state"])
        except Exception:
            pass
        return ckpt
    except Exception:
        return None


def _run_training_loop(  # noqa: C901
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
            # Move the input tensor to the agent's device so model and data
            # reside on the same device (important for GPU/Colab usage).
            state_tensor = (
                torch.tensor(np.array(state), dtype=torch.float32)
                .unsqueeze(0)
                .to(agent.device)
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
            # Ensure the action tensor is on the same device as the model/distribution
            log_probs.append(
                dist.log_prob(torch.tensor(action, device=agent.device)).item()
            )
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

        # Compute a bootstrapped last value; move inputs to agent.device first.
        last_value = (
            0
            if done
            else agent.model(
                torch.tensor(np.array(state), dtype=torch.float32)
                .unsqueeze(0)
                .to(agent.device)
            )[1].item()
        )
        values.append(last_value)
        advantages = agent.compute_advantages(
            np.array(rewards), np.array(values), np.array(dones)
        )
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

        print(
            f"Iter {episode + 1}/{episodes} | env_eps: {env_episodes_this_iter} | Avg100: {avg_reward}"
        )

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(agent.model.state_dict(), save_path)
            # Also save a full checkpoint (model + optimizers + RNGs) so we can
            # resume training exactly from this state.
            try:
                checkpoint_path = os.path.join(
                    os.path.dirname(save_path) or ".", "checkpoint.pt"
                )
                _save_checkpoint(checkpoint_path, agent, episode + 1, best_avg_reward)
            except Exception:
                pass

        try:
            if latest_interval > 0 and ((episode + 1) % latest_interval) == 0:
                latest_path = os.path.join(
                    os.path.dirname(save_path) or ".", "latest_model.pt"
                )
                os.makedirs(os.path.dirname(latest_path), exist_ok=True)
                torch.save(agent.model.state_dict(), latest_path)
                # Also write a full checkpoint for safer resumes
                try:
                    checkpoint_path = os.path.join(
                        os.path.dirname(save_path) or ".", "checkpoint.pt"
                    )
                    _save_checkpoint(
                        checkpoint_path, agent, episode + 1, best_avg_reward
                    )
                except Exception:
                    pass
        except Exception:
            pass

        save_state(state_path, episode + 1, best_avg_reward)

    return all_rewards, best_avg_reward


def main(  # noqa: C901
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
                print(
                    f"Warning: loaded best_avg_reward={loaded_best} invalid; resetting to -inf."
                )
                best_avg_reward = -float("inf")
            else:
                best_avg_reward = loaded_best
        except Exception:
            episodes_completed = 0
    # If we're resuming from a previous run, attempt to load a full checkpoint
    # (model + optimizers + RNGs). If a full checkpoint is not available, fall
    # back to loading model weights from `latest_model.pt` or `save_path` so we
    # don't reinitialize the network and experience a drop in performance.
    if episodes_completed > 0:
        checkpoint_path = os.path.join(
            os.path.dirname(save_path) or ".", "checkpoint.pt"
        )
        loaded_meta = None
        if os.path.exists(checkpoint_path):
            loaded_meta = _load_checkpoint(checkpoint_path, agent)
            if loaded_meta is not None:
                print(
                    f"Resuming training: loaded full checkpoint from {checkpoint_path}"
                )
        if loaded_meta is None:
            latest_path = os.path.join(
                os.path.dirname(save_path) or ".", "latest_model.pt"
            )
            ckpt_to_load = None
            if os.path.exists(latest_path):
                ckpt_to_load = latest_path
            elif os.path.exists(save_path):
                ckpt_to_load = save_path
            if ckpt_to_load is not None:
                try:
                    agent.model.load_state_dict(
                        torch.load(ckpt_to_load, map_location=agent.device)
                    )
                    print(
                        f"Resuming training: loaded model weights from {ckpt_to_load}"
                    )
                except Exception as e:
                    print(
                        f"Warning: failed to load model checkpoint '{ckpt_to_load}': {e}"
                    )
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
