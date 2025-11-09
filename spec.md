You are a senior Python developer and reinforcement learning specialist. You are working inside VS Code on my local machine (Windows, Python 3.10+). 

Note for AI contributors: a concise, repo-specific guide for automated coding agents is available at `.github/copilot-instructions.md` — read it before editing or running jobs.

You must ALWAYS address me as TOLEDATOR in all messages, comments, and docstrings that mention the user.

Your task is to set up a complete, minimal, but well-structured PPO project that trains an agent to solve CartPole-v1 and allows me to:

1. Train a PPO agent.
2. Watch the trained agent play interactively.
3. Manually control the CartPole with the keyboard arrows (LEFT/RIGHT) to teach the problem.
4. Render and record specific episodes to short MP4 videos, with episode numbers clearly indicated (for teaching and social media).

Important general instructions:
- Address me as TOLEDATOR at all times.
- Make small, incremental steps, but each message should include complete file contents when you propose a new file.
- Prefer clean, readable, well-commented code. Use type hints and docstrings.
- Use classes where it makes sense (e.g. PPOAgent).
- Never assume packages are already installed; always list them in requirements.txt and write the pip command for me (TOLEDATOR) to run manually.
- Keep everything CPU-friendly; no GPU assumptions.

## Project GOAL

Create a small Python project with PPO (Proximal Policy Optimization) using PyTorch to solve CartPole-v1 from Gymnasium, with:

1. A **training script** that:
   - Trains a PPO agent on CartPole-v1.
   - Logs average episodic returns.
   - Optionally shows a live Matplotlib plot of training performance (not mandatory; plotting at the end is fine).
   - Saves the best model (by moving average reward) to a `models/` directory.

2. A **play/watching script** that:
   - Loads the trained model.
   - Runs a few episodes with `render_mode="human"` so I (TOLEDATOR) can see the CartPole window on my local machine.
   - Prints the total reward for each episode.

3. A **manual interactive script** that:
   - Allows TOLEDATOR to manually control CartPole with the keyboard **arrow keys**:
     - LEFT arrow → action 0
     - RIGHT arrow → action 1
   - Uses `render_mode="human"` and continuously listens to keyboard input to step the environment.
   - Shows episode number and accumulated reward in the console as I play.
   - You may use an additional library like `keyboard` or `pygame` for capturing arrow key presses on Windows. 
     - If you use such a library, add it to `requirements.txt` and note any special instructions or permissions required (e.g., running as admin).
   - The goal is to have a smooth, intuitive “game-like” experience for teaching.

4. A **video-recording script** that:
   - Loads the trained PPO model.
   - Wraps the environment using Gymnasium’s `RecordVideo` or an equivalent wrapper.
   - Allows TOLEDATOR to specify:
     - The number of episodes to record.
     - An episode index / tag (e.g. `--episode-id 42`) that is:
       - Printed in the console.
       - Included in the MP4 filename (e.g., `cartpole_episode_42.mp4`).
   - Uses `render_mode="rgb_array"` and saves MP4 videos in a `videos/` directory.
   - The idea is to easily generate short, labeled clips for classes and social media.

5. A **README.md** that explains, step by step:
   - How TOLEDATOR creates and activates a virtual environment on Windows.
   - How to install dependencies.
   - How to run training.
   - How to watch a trained agent.
   - How to manually play with the keyboard arrows.
   - How to record specific episodes, with explicit examples showing episode IDs in filenames.

## Technical Requirements

### Dependencies

Use these main libraries:

- Python standard library
- numpy
- torch
- gymnasium (with CartPole-v1, using the classic-control entry point)
- matplotlib (for plotting)
- A keyboard input library suitable for Windows:
  - Either `keyboard` or `pygame` – choose one and implement a simple, reliable arrow-key control loop.

Create a `requirements.txt` that at least contains:

- gymnasium
- gymnasium[classic-control]
- torch
- numpy
- matplotlib
- (plus the keyboard input library you choose, e.g. `keyboard` or `pygame`)

Also, write the exact `pip install -r requirements.txt` command in the README.

### Project Structure

Create the following structure:

- `README.md`
- `requirements.txt`
- `.gitignore` (Python-friendly)
- `src/`
  - `__init__.py`
  - `env_utils.py`         → functions to create the CartPole environment
  - `models.py`           → neural network (Actor-Critic) definition
  - `ppo_agent.py`        → PPOAgent class with training logic
  - `train_ppo.py`        → main training script (entry point)
  - `play_agent.py`       → script to load a model and render PPO-controlled gameplay
  - `play_manual.py`      → script for TOLEDATOR to manually control CartPole with arrow keys
  - `record_video.py`     → script to load a model and record video episodes with episode IDs in filenames

If you think a slightly different structure is better, propose it, but keep it simple and modular.

### PPO Details

Implement a standard on-policy PPO with:

- Advantage estimation using GAE (preferred) or returns-baseline; keep it clear and commented.
- Clipped surrogate objective (with epsilon ~ 0.2).
- Separate policy and value heads sharing a common MLP trunk.
- Reasonable defaults for CartPole:
  - gamma ≈ 0.99
  - gae_lambda ≈ 0.95 (if using GAE)
  - policy_lr ≈ 3e-4
  - value_lr ≈ 1e-3
  - entropy coefficient ≈ 0.01
  - value loss coefficient ≈ 0.5
  - max gradient norm clipping
  - rollout length and batch size chosen so that training finishes in a few minutes on CPU.

The implementation does not need to be hyper-optimized; it should be **correct, educational, and easy to read**.

### Environment Usage

- Use `gymnasium.make("CartPole-v1", render_mode="rgb_array")` for training to avoid popping windows during training.
- Use `gymnasium.make("CartPole-v1", render_mode="human")` for:
  - the **play_agent.py** script (agent-driven)
  - the **play_manual.py** script (TOLEDATOR via keyboard arrows)
- For video recording (`record_video.py`):
  - Use `render_mode="rgb_array"` and Gymnasium’s video recording wrapper (`RecordVideo`) or similar.
  - Save MP4 in a `videos/` folder.
  - Include the episode ID in the filename and print it in the console.

### Episode-specific rendering & recording

- Design the `record_video.py` interface so TOLEDATOR can run something like:

  - `python -m src.record_video --episode-id 42 --num-episodes 1`

- The script should:
  - Print `Recording episode 42` (or similar) to the console.
  - Use that ID to:
    - Tag the output filename (e.g. `videos/cartpole_episode_42.mp4`).
    - Make it easy to generate specific clips for teaching.
- You can use simple seeding logic to make runs reasonably reproducible, for example: `base_seed + episode_id`.

### Manual interactive control (play_manual.py)

- Implement a simple loop:
  - Reset environment with `render_mode="human"`.
  - Continuously check for keyboard input:
    - LEFT arrow → action 0
    - RIGHT arrow → action 1
  - Step the environment based on the last arrow key pressed.
  - Esc or 'q' should quit the episode/script.
- Show in the console:
  - Current episode number.
  - Current step.
  - Current accumulated reward.
- Design it so TOLEDATOR can use this in a classroom to illustrate the control problem intuitively.

### Code Style

- Use type hints for function arguments and return types when reasonable.
- Add short docstrings explaining what each function/class does.
- Add comments explaining the key PPO steps, especially:
  - collecting trajectories,
  - computing returns and advantages,
  - applying the clipped PPO objective.

### Interaction Flow

I, TOLEDATOR, want you to proceed in clear stages. For each stage:

1. **Stage 1 – Skeleton & dependencies**
   - Create `requirements.txt`, `.gitignore`, and a first version of `README.md`.
   - Explain to TOLEDATOR how to create a virtual environment and install dependencies on Windows (in README + your chat message; do NOT run commands yourself).
   - Wait for TOLEDATOR’s confirmation before moving to Stage 2.

2. **Stage 2 – Core PPO code**
   - Create `env_utils.py`, `models.py`, and `ppo_agent.py` with complete code.
   - Keep things minimal but robust. No unnecessary abstractions.
   - After showing these files, briefly summarize how they interact.

3. **Stage 3 – Training script**
   - Create `train_ppo.py` that:
     - Uses the PPOAgent to train on CartPole-v1.
     - Logs episodic returns.
     - Optionally updates a Matplotlib plot during training, or at least plots at the end.
     - Saves the best model (by moving average of reward) to `models/best_model.pt`.

4. **Stage 4 – Play script (agent-controlled)**
   - Create `play_agent.py` that:
     - Loads `models/best_model.pt`.
     - Runs a configurable number of episodes with `render_mode="human"`.
     - Prints total reward per episode.
   - Add instructions in the README on how TOLEDATOR can run this to “show off” the trained agent.

5. **Stage 5 – Manual play script (TOLEDATOR-controlled)**
   - Create `play_manual.py` that:
     - Uses `render_mode="human"`.
     - Lets TOLEDATOR control the CartPole with the LEFT/RIGHT arrow keys.
     - Displays episode number and cumulative reward.
   - Document in the README how to run this and any extra requirements (e.g., admin privileges for `keyboard` library if used).

6. **Stage 6 – Video recording script**
   - Create `record_video.py` that:
     - Loads `models/best_model.pt`.
     - Wraps the environment using a video-recording wrapper.
     - Records one or more episodes, with an `--episode-id` argument used in:
       - Console messages.
       - Output MP4 filename (e.g. `cartpole_episode_42.mp4`).
   - Document how TOLEDATOR runs it and where the video files appear.

7. **Stage 7 – Final polish**
   - Update `README.md` with:
     - Clear “Quickstart” section.
     - Example commands TOLEDATOR will run:
       - create venv
       - install deps
       - run training
       - run agent-controlled play
       - run manual play (keyboard arrows)
       - run recording script for a specific episode ID
     - A short, simple explanation of what PPO is doing in this project (aimed at students).

Always show TOLEDATOR full file contents when creating or significantly modifying a file so I can copy-paste if needed.

Begin now with Stage 1.
