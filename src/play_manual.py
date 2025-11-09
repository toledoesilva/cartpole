"""
Script for manual control of the CartPole environment using keyboard arrows.
"""

import gymnasium as gym
import keyboard
import time  # Add this import for introducing delays

# Use the original CartPole environment with extended time limit
env = gym.make("CartPoleExtended-v1", render_mode="human")

print("Use LEFT and RIGHT arrow keys to control the CartPole.")
print("Press ESC or 'q' to quit.")

# Default action when no key is pressed
default_action = 1  # Example: Move right

while True:
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = default_action  # Default action if no key is pressed

        # Check for keyboard input
        if keyboard.is_pressed("left"):
            action = 0
        elif keyboard.is_pressed("right"):
            action = 1
        elif keyboard.is_pressed("esc") or keyboard.is_pressed("q"):
            print("Exiting...")
            env.close()
            exit()

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

        time.sleep(0.05)  # Add a 50ms delay to slow down the game

    print(f"Episode finished. Total Reward: {total_reward}")

env.close()
