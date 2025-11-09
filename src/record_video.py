"""Record episodes with the trained PPO model and save them as MP4 files.

This script supports Gymnasium/Gym API differences and renames the generated
video to a predictable path `videos/cartpole_episode_<id>.mp4` so it's easy to
find after recording.
"""

import argparse
import glob
import os
import shutil
import time
from typing import Any, Tuple

import numpy as np
import torch
from src.env_utils import create_environment, wrap_for_recording
from src.models import ActorCritic


def _unwrap_reset(reset_ret: Any):
    if isinstance(reset_ret, tuple) or isinstance(reset_ret, list):
        return reset_ret[0]
    return reset_ret


def _unwrap_step(step_ret: Tuple):
    if len(step_ret) == 5:
        next_state, reward, terminated, truncated, info = step_ret
        done = bool(terminated or truncated)
        return next_state, reward, done, info
    if len(step_ret) == 4:
        next_state, reward, done, info = step_ret
        return next_state, reward, bool(done), info
    raise ValueError(f"Unexpected step return signature: {step_ret}")


def main(
    episode_id: int,
    num_episodes: int = 1,
    model_path: str = "models/best_model.pt",
    video_dir: str = "videos",
):
    os.makedirs(video_dir, exist_ok=True)
    # Load trained model (map to CPU if needed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We'll record each episode in its own wrapped env so we can map files to
    # episode rewards reliably.
    recorded_files = []
    rewards = []

    for ep in range(num_episodes):
        # Create and wrap environment for this single episode
        env = create_environment(render_mode="rgb_array")
        env = wrap_for_recording(env, video_dir)

        input_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Load trained model for this episode
        model = ActorCritic(input_dim, action_dim).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        reset_ret = env.reset(seed=episode_id + ep)
        state = _unwrap_reset(reset_ret)
        total_reward = 0.0
        done = False

        while not done:
            state_tensor = (
                torch.tensor(np.array(state), dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                logits, _ = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()

            step_ret = env.step(action)
            next_state, reward, done, _info = _unwrap_step(step_ret)
            total_reward += float(reward)
            state = next_state

        print(f"Recorded Episode {episode_id + ep}: Total Reward = {total_reward}")

        # Close env to flush recording files for this episode
        env.close()
        time.sleep(0.2)  # let IO settle

        # Find the most recent mp4 file and associate it with this episode
        mp4s = sorted(
            glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True),
            key=os.path.getmtime,
        )
        if mp4s:
            latest = mp4s[-1]
            recorded_files.append(latest)
            rewards.append(total_reward)
        else:
            recorded_files.append(None)
            rewards.append(total_reward)

    # Select best episode by reward
    if not any(recorded_files):
        print(f"No mp4 files found under {video_dir}")
        return

    # choose the highest reward; if multiple share the same max, pick the latest
    rewards_arr = np.array(rewards)
    max_reward = rewards_arr.max()
    best_idxs = np.where(rewards_arr == max_reward)[0]
    best_idx = int(best_idxs[-1])
    best_file = recorded_files[best_idx]
    best_episode = episode_id + best_idx
    target = os.path.join(video_dir, f"cartpole_best_episode_{best_episode}.mp4")

    try:
        shutil.copy2(best_file, target)
        print(
            f"Saved best video (episode {best_episode}, reward={rewards[best_idx]}) to {target}"
        )
    except Exception as e:
        print(f"Failed to copy best video: {e}. Best file: {best_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record CartPole episodes with a trained PPO model."
    )
    parser.add_argument(
        "--episode-id", type=int, required=True, help="Episode ID to tag the recording."
    )
    parser.add_argument(
        "--num-episodes", type=int, default=1, help="Number of episodes to record."
    )
    args = parser.parse_args()
    main(args.episode_id, args.num_episodes)
