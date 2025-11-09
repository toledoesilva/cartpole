"""Play a trained PPO model and render agent-controlled gameplay.

This script is defensive about Gym / Gymnasium API differences and maps the
loaded model to CPU if needed. It uses `create_environment` from
`src.env_utils` and expects the model at `models/best_model.pt` by default.
"""

import torch
import numpy as np
from typing import Any, Tuple
from src.env_utils import create_environment
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


def main(model_path: str = "models/best_model.pt", num_episodes: int = 5):
    env = create_environment(render_mode="human")
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load trained model (map to CPU if GPU not available or model was saved on GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(input_dim, action_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    for episode in range(num_episodes):
        reset_ret = env.reset()
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

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
