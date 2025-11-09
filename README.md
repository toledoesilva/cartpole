# CartPole PPO Project

This project implements a Proximal Policy Optimization (PPO) agent to solve the CartPole-v1 environment from Gymnasium. The implementation is designed to be modular, educational, and CPU-friendly.

## Quickstart

### 1. Create a Virtual Environment
Run the following commands in PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
Install the required Python packages:
```powershell
pip install -r requirements.txt
```

### 3. Train the PPO Agent
To train the PPO agent, run:
```powershell
python -m src.train_ppo
```
This will log episodic rewards and save the best model to `models/best_model.pt`.

### 4. Watch the Trained Agent
To watch the trained agent play, run:
```powershell
python -m src.play_agent
```

### 5. Manually Control the CartPole
To manually control the CartPole using keyboard arrows, run:
```powershell
python -m src.play_manual
```
Use the LEFT/RIGHT arrow keys to control the CartPole. Press `Esc` or `q` to quit.

### 6. Record Episodes
To record specific episodes, run:
```powershell
python -m src.record_video --episode-id 42 --num-episodes 1
```
This will save the video to `videos/cartpole_episode_42.mp4`.

## PPO Overview
Proximal Policy Optimization (PPO) is a reinforcement learning algorithm that balances exploration and exploitation by optimizing a clipped surrogate objective. This project uses:
- Generalized Advantage Estimation (GAE) for advantage computation.
- Separate policy and value networks sharing a common backbone.
- Hyperparameters tuned for CartPole-v1.

## Setup Instructions

### 1. Create a Virtual Environment
Run the following commands in PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
Install the required Python packages:
```powershell
pip install -r requirements.txt
```

## Usage

### Training the Agent
To train the PPO agent, run:
```powershell
python -m src.train_ppo
```

### Watching the Agent Play
To watch the trained agent, run:
```powershell
python -m src.play_agent
```

### Manual Control
To manually control the CartPole using keyboard arrows, run:
```powershell
python -m src.play_manual
```

### Recording Videos
To record specific episodes, run:
```powershell
python -m src.record_video --episode-id 42 --num-episodes 1
```

This will save the video to `videos/cartpole_episode_42.mp4`.

## AI contributor notes
If you're an automated agent or contributor, read the repository-specific agent guidance before editing:

- The concise agent instructions live at `.github/copilot-instructions.md` and list the key files, CLI checks, and project-specific rules (CPU-first, model artifact paths, video outputs).
- Quick PowerShell checks:
```powershell
python -m venv venv; .\venv\Scripts\activate
pip install -r requirements.txt
python -m src.train_ppo --help
python -m src.record_video --episode-id 0 --num-episodes 1
```

Read `.github/copilot-instructions.md` before making non-trivial changes (hyperparams, model serialization, render modes, or dependencies).