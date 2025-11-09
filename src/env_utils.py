"""
Utility functions for creating and managing the CartPole environment.
"""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.registration import register

from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set custom termination conditions
        self.theta_threshold_radians = 45 * (3.141592 / 180)  # 45 degrees in radians
        self.x_threshold = 2.4  # Revert to default x_threshold to ensure the cart stays within the screen

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        # Override termination condition to include custom thresholds
        done = bool(
            self.state[0] < -self.x_threshold
            or self.state[0] > self.x_threshold
            or self.state[2] < -self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
        )
        return observation, reward, done, truncated, info

    def render(self, *args, **kwargs):
        # Override render to maintain the original pole length visually
        original_x_threshold = 2.4  # Default x_threshold in the original CartPoleEnv

        # Temporarily adjust x_threshold for rendering
        original_threshold = self.x_threshold
        self.x_threshold = original_x_threshold
        rendered_frame = super().render(*args, **kwargs)
        self.x_threshold = original_threshold

        return rendered_frame


# Update the registration to use the custom environment
register(
    id="CartPoleExtended-v1",
    entry_point="src.env_utils:CustomCartPoleEnv",
    max_episode_steps=10_000,  # Extended time limit to 10,000 steps
)


def create_environment(render_mode: str = "rgb_array") -> gym.Env:
    """
    Create the CartPole-v1 environment with the specified render mode.

    Args:
        render_mode (str): The render mode for the environment. Options are "rgb_array" or "human".

    Returns:
        gym.Env: The created CartPole environment.
    """
    # Use the registered custom environment id
    return gym.make("CartPoleExtended-v1", render_mode=render_mode)


def wrap_for_recording(env: gym.Env, video_dir: str) -> gym.Env:
    """
    Wrap the environment for video recording.

    Args:
        env (gym.Env): The environment to wrap.
        video_dir (str): Directory to save recorded videos.

    Returns:
        gym.Env: The wrapped environment.
    """
    # RecordVideo in gymnasium expects `video_folder` keyword
    return RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True)
