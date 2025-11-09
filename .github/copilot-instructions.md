## Copilot instructions — CartPole PPO (concise)

Short, actionable guidance for AI contributors working on this repo.

Read these files first (order matters):
- `src/env_utils.py` — registers `CartPoleExtended-v1`, provides `create_environment(render_mode)` and `wrap_for_recording()`; prefer `render_mode='rgb_array'` for recordings and `'human'` for interactive play.
- `src/models.py` — ActorCritic network: shared trunk (128,128) and two heads (policy logits, value scalar).
- `src/ppo_agent.py` — PPO implementation (GAE, clipped objective). Policy and value use separate optimizers.
- `src/train_ppo.py` — CLI entrypoint, training loop, checkpoint logic, `training_state.json` usage and model save locations (`models/best_model.pt`, `models/latest_model.pt`, `checkpoint.pt`).
- `src/record_video.py` — recording pattern: create per-episode wrapped envs, record frames, and copy the best MP4 into `videos/`.
- `src/play_agent.py` / `src/play_manual.py` — scripted playback vs manual keyboard control (may require `keyboard` or `pygame`).

Quick PowerShell checks (safe, fast validation):
```powershell
python -m venv venv; .\venv\Scripts\activate
pip install -r requirements.txt
python -m src.train_ppo --help
python -m src.record_video --episode-id 0 --num-episodes 1
python -m src.play_agent
```

Project-specific conventions and gotchas:
- CPU-first: defaults are CPU-friendly. Avoid GPU-only changes unless requested. Code maps models to device using `torch.device("cuda" if available else "cpu")`.
- Model artifacts: preserve `models/best_model.pt` and `models/latest_model.pt`. `checkpoint.pt` stores model+optimizers+RNGs for exact resumes.
- Training resume: `training_state.json` tracks episodes completed and best_avg_reward; `train_ppo.py` tries `checkpoint.pt` first, then `latest_model.pt` or `best_model.pt`.
- Env id: `CartPoleExtended-v1` is registered with an extended max-steps limit. Use `create_environment()` to stay consistent across scripts.
- API compatibility: use `_unwrap_reset` and `_unwrap_step` helpers (present in multiple scripts) to support both Gym and Gymnasium call signatures.

Concrete, copyable examples:
- Add an eval CLI: copy `src/record_video.py` — create an env with `render_mode='rgb_array'`, call `wrap_for_recording()`, run an episode, and save the resulting MP4 in `videos/` with predictable naming.
- Change network architecture: edit `src/models.py` (shared trunk + two heads). Validate by running a short training: `python -m src.train_ppo --episodes 2 --rollout-length 16`.
- Update hyperparameters: change defaults in both `src/ppo_agent.py` and `src/train_ppo.py` (implementation and CLI docs must match).

Integration points & dependencies to watch:
- Primary: `gymnasium`, `torch`, `numpy`, `matplotlib` (see `requirements.txt`). Optional: `pygame` or `keyboard` for manual play.
- IO: videos under `videos/`, training state `training_state.json`, and several historical backups may exist in the repo root.

Agent-edit contract (inputs / outputs / checks):
- Input: small, self-contained edits in `src/` (training, eval, utils). Prefer minimal-scope PRs.
- Output: must remain runnable with `python -m src.train_ppo` on CPU. If you change CLI defaults, update help text in `src/train_ppo.py`.
- Smoke tests: run `python -m src.train_ppo --help` and `python -m src.record_video --episode-id 0 --num-episodes 1`.

Quality gates and quick verification:
- Lint: `black` and `ruff` are listed in `requirements.txt` — run them when changing style.
- Runtime sanity: short training run (2 episodes, small rollout) to catch shape/serialization or device errors and to ensure model artifacts are saved.

If you'd like the file expanded with sample PR templates, unit-test guidance, or a short debugging checklist (how to inspect `checkpoint.pt` / RNG state), tell me which section to expand and I'll update this file.
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