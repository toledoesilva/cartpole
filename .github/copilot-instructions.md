## Copilot instructions — CartPole PPO (concise)

Short, actionable guidance for AI contributors working on this repo (CartPole PPO).

Read these key files first:
- `src/env_utils.py` — env creation and `render_mode` choices (`'rgb_array'` for recording/eval, `'human'` for interactive play).
- `src/models.py` — actor-critic network (shared trunk, two heads).
- `src/ppo_agent.py` — PPO training loop, GAE, clipped objective and core hyperparams.
- `src/train_ppo.py` — CLI entrypoint (default hyperparams) and model saving (`models/best_model.pt`).
- `src/record_video.py` — example for running envs with `render_mode='rgb_array'` and saving MP4s to `videos/`.
- `src/play_agent.py` / `src/play_manual.py` — loading `models/best_model.pt` and interactive controls (uses `keyboard` or `pygame`).

Quick PowerShell commands (use these to test or run):
```powershell
python -m venv venv; .\venv\Scripts\activate
pip install -r requirements.txt
python -m src.train_ppo         # train and produce models/best_model.pt
python -m src.train_ppo --help # inspect CLI before long runs
python -m src.play_agent       # load best model and render
python -m src.play_manual      # manual play (arrow keys)
python -m src.record_video --episode-id 1 --num-episodes 1
```

Project-specific rules and patterns (do not bypass):
- Keep training CPU-friendly by default (small MLPs, no CUDA-only changes) unless the user asks for GPU support.
- Model artifact locations: `models/best_model.pt` and `models/latest_model.pt`. Preserve these filenames unless you add a CLI flag that explicitly changes output paths.
- Video outputs: write to `videos/` and include the episode id in filename (see `record_video.py`).
- When changing hyperparameters, update both `src/ppo_agent.py` (implementation) and `src/train_ppo.py` (CLI defaults) together.

Concrete editing examples:
- Add a new eval mode: copy the pattern from `src/record_video.py` — create env with `render_mode='rgb_array'`, collect frames, and write MP4 to `videos/`.
- Change network architecture: modify `src/models.py` and run a short training (small num_steps) to verify no shape/serialization breakage.
- Add keyboard input support: follow `src/play_manual.py` to detect whether `keyboard` or `pygame` is used and add dependency to `requirements.txt` if needed.

Integration and external deps to watch for:
- `gymnasium`, `torch`, `numpy`, and `matplotlib` are primary dependencies (see `requirements.txt`).
- Manual play can depend on `keyboard` or `pygame` — check `src/play_manual.py` before adding code that assumes one or the other.

Agent-edit contract (inputs, outputs, checks):
- Input: small, self-contained edits in `src/` (training, eval, utils).
- Output: must remain runnable with `python -m src.train_ppo` on CPU. If you change defaults, update `train_ppo.py`'s CLI help.
- Quick checks: run `python -m src.train_ppo --help` and a brief `python -m src.record_video --episode-id 0 --num-episodes 1` to validate changes.

Quality and safety notes:
- Avoid adding GPU-only code or heavy dependencies unless requested.
- Preserve the explicit `render_mode` usage in `env_utils.py` to keep recording stable.

If you want the file to be more/less detailed or prefer a different structure for agent instructions, tell me which parts to expand or trim.