# An Agent That Knows When to Ask

**Can an autonomous agent recognise when it is unsure, and hand control to a human at those moments?**

This project studies that question in a driving simulator. It is an early-stage, ongoing research project.

## Motivation

I build an LLM-based agent that designs Piping and Instrumentation Diagrams (P&IDs) for chemical plants. A key question for that agent is when to decide on its own and when to ask the engineer. Engineering design gives slow and expensive feedback, so the question is hard to study there directly.

A driving simulator gives fast, measurable feedback on every decision (did the car crash or not?). This project uses it to develop and test a "defer when unsure" method, which I then plan to bring back to the P&ID agent.

## Approach

1. **Baselines:** train model-free agents (DQN and PPO) to drive in the [`highway-env`](https://github.com/Farama-Foundation/HighwayEnv) simulator.
2. **World-model agent:** train a Dreamer-style agent that learns a model of how traffic evolves and learns to drive inside that model.
3. **Uncertainty signal:** add several small predictors on top of the world model and measure how much their predictions disagree. High disagreement means the agent is in an unfamiliar situation.
4. **Deferring:** when disagreement is above a threshold, a cautious backup driver takes control for that step.
5. **Evaluation:** train on normal traffic only, then test on traffic the agent has never seen. Plot crash rate against how often the agent asks for help, and compare against two simple references:
   - never deferring
   - deferring at random moments, equally often

   If disagreement-based deferring is safer than random deferring at the same help rate, the uncertainty signal is detecting the dangerous moments.

## Current results: model-free baselines

DQN and PPO, each trained for 50,000 steps on `highway-fast-v0` with 3 random seeds, then tested on the same 100 fixed test roads (test seeds 10000–10099, never used in training). Test-time actions are deterministic. Numbers are mean ± standard deviation across seeds.

| Agent  | Seeds | Crash rate          | Return       | Distance (m) | Speed (m/s) |
|--------|-------|---------------------|--------------|--------------|-------------|
| Random | 1     | 97% ± – (97%–97%)   | 6.86 ± –     | 224          | 24.1        |
| DQN    | 3     | 40% ± 16 (22–52%)   | 22.60 ± 1.45 | 693          | 27.5        |
| PPO    | 3     | 25% ± 9 (14–32%)    | 21.25 ± 0.76 | 611          | 22.5        |

**What this shows**

- Both agents learn to drive far better than random.
- The two algorithms learn different styles: DQN drives fast and earns slightly more reward; PPO drives slower and crashes less on average. Every seed follows this pattern.
- Results vary a lot between seeds (DQN: 22% to 52% crashes), so single runs are not reliable.
- Even the best agents crash in roughly 1 of 4 episodes. These risky moments are what the deferral method is designed to catch.

Per-run results are in [`results/eval.csv`](results/eval.csv) (one row per test episode) and [`results/summary.csv`](results/summary.csv).

### How they drive

Same test road (test seed 10000), seed-0 agent of each algorithm:

| Random (for reference) | DQN (fast, riskier) | PPO (slower, more cautious) |
|---|---|---|
| ![Random agent driving](assets/random.gif) | ![DQN driving](assets/dqn.gif) | ![PPO driving](assets/ppo.gif) |

*One episode each, for illustration only; the table above is based on 100 test roads per agent.*

### Design choices

- **Training environment:** `highway-fast-v0` instead of `highway-v0`. It is the same task (same observations, actions and reward) with fewer cars and coarser physics, and it simulates about 10× faster (measured: 60 vs. 5.9 steps per second). This makes a multi-seed study with a world-model agent feasible on a laptop GPU. `highway-v0` will be used as an additional test condition.
- **Evaluation:** every agent drives the same 100 fixed test roads; each episode is stored as one row; crash rates are reported with 95% confidence intervals per run and with the spread across seeds per algorithm.

## Roadmap

- [x] Set up `highway-env`; DQN and PPO baselines with 3 seeds each
- [x] Shared evaluation: 100 fixed test roads, per-episode results, summary across seeds
- [ ] Train a Dreamer-style world-model agent (DreamerV3)
- [ ] Add disagreement heads and check that disagreement rises before crashes
- [ ] Implement deferring to a backup driver, with thresholds set on familiar traffic only
- [ ] Evaluate on unseen traffic (denser, aggressive drivers, fewer lanes, merge, `highway-v0`): crash rate vs. help-request rate, compared against random deferring
- [ ] Write up the results

## Setup

1. Install PyTorch for your system first, using the command from [pytorch.org](https://pytorch.org/get-started/locally/) (pick a CUDA build if you have an NVIDIA GPU).
2. Install the other packages:

```
pip install -r requirements.txt
```

To reproduce the exact package versions used for the results, use `requirements-lock.txt` instead (PyTorch is not listed there; install it as in step 1).

## How to run

Run everything from the repository's top folder.

```
# check the install and watch a random agent drive
python scripts/check_setup.py

# train the baselines (one run per seed)
python -m baselines.train_dqn --seed 0 --steps 50000
python -m baselines.train_ppo --seed 0

# evaluate on 100 fixed test roads (appends to results/eval.csv)
python -m eval.run_eval --agent random
python -m eval.run_eval --agent dqn --model models/dqn_seed0 --train-seed 0 --gif assets/dqn.gif
python -m eval.run_eval --agent ppo --model models/ppo_seed0 --train-seed 0 --gif assets/ppo.gif

# summary tables across seeds
python -m eval.summarize
```

Training logs can be viewed with `tensorboard --logdir logs/`.

## Repository structure

| Path | What it contains |
|------|------------------|
| `envs/conditions.py` | Creates every environment used in the project (training and test conditions) |
| `baselines/train_dqn.py` | Trains the DQN baseline (Stable-Baselines3) |
| `baselines/train_ppo.py` | Trains the PPO baseline (Stable-Baselines3) |
| `eval/run_eval.py` | Tests any agent on fixed test roads; one CSV row per episode |
| `eval/summarize.py` | Summary tables per run and per algorithm across seeds |
| `scripts/check_setup.py` | Checks the installation and runs a random agent |
| `scripts/speed_test.py` | Compares simulation speed of `highway-fast-v0` and `highway-v0` |
| `results/` | Evaluation results (CSV) |
| `assets/` | GIFs and figures used in this README |

Trained models (`models/`) and training logs (`logs/`) are not stored in the repository.

## Built on

- [highway-env](https://github.com/Farama-Foundation/HighwayEnv): the driving simulator
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3): the DQN and PPO implementations
- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3): the world-model approach this project builds on

## License

MIT, see [LICENSE](LICENSE).