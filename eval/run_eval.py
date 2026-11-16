"""
Evaluate any agent on fixed test roads and append one row per episode to a CSV.

Run from the repo's top folder, for example:
    python -m eval.run_eval --agent random
    python -m eval.run_eval --agent dqn --model models/dqn_seed0 --train-seed 0
    python -m eval.run_eval --agent ppo --model models/ppo_seed0 --train-seed 0 --gif assets/ppo.gif

Every agent faces the SAME test roads (test seeds 10000, 10001, ...),
which are never used in training. Results go to results/eval.csv.
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

from envs.conditions import make_env

TEST_SEED_START = 10_000

parser = argparse.ArgumentParser()
parser.add_argument("--agent", required=True, choices=["random", "dqn", "ppo"])
parser.add_argument("--model", default=None, help="path to a saved model (not needed for random)")
parser.add_argument("--train-seed", type=int, default=-1, help="which training run this model came from")
parser.add_argument("--condition", default="normal")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--out", default="results/eval.csv")
parser.add_argument("--gif", default=None, help="save the first test episode as a GIF")
args = parser.parse_args()


def load_policy(agent, model_path):
    """Returns (policy function, number of training steps the model was trained for)."""
    if agent == "random":
        return (lambda obs, env: env.action_space.sample()), 0
    if model_path is None:
        raise ValueError("--model is required for trained agents")
    if agent == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(model_path, device="cpu")
    else:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")
    policy = lambda obs, env: int(model.predict(obs, deterministic=True)[0])
    return policy, int(model.num_timesteps)   # SB3 stores how many steps the model was trained for


def wilson_interval(k, n, z=1.96):
    """95% confidence interval for a rate k/n (works well even near 0% or 100%)."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


policy, train_steps = load_policy(args.agent, args.model)
env = make_env(args.condition, render_mode="rgb_array" if args.gif else None)

rows, frames = [], []
for ep in range(args.episodes):
    test_seed = TEST_SEED_START + ep
    obs, info = env.reset(seed=test_seed)
    env.action_space.seed(test_seed)   # makes random actions repeatable too
    ego = env.unwrapped.vehicle
    x_start = float(ego.position[0])
    done = truncated = False
    total_reward, steps, speeds = 0.0, 0, []

    while not (done or truncated):
        action = policy(obs, env)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        speeds.append(float(env.unwrapped.vehicle.speed))
        if args.gif and ep == 0:
            frames.append(env.render())

    rows.append({
        "agent": args.agent,
        "model": args.model or "none",
        "train_steps": train_steps,
        "train_seed": args.train_seed,
        "condition": args.condition,
        "episode": ep,
        "test_seed": test_seed,
        "crashed": bool(info["crashed"]),
        "return": total_reward,
        "steps": steps,
        "distance_m": float(env.unwrapped.vehicle.position[0]) - x_start,
        "mean_speed_mps": float(np.mean(speeds)),
        "help_requests": 0,          # used from Week 5 (deferral)
    })
env.close()

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
df.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)

if args.gif:
    import imageio
    os.makedirs(os.path.dirname(args.gif), exist_ok=True)
    imageio.mimsave(args.gif, frames, duration=0.3)

k, n = int(df["crashed"].sum()), len(df)
lo, hi = wilson_interval(k, n)
print(f"\n{args.agent} (train seed {args.train_seed}, {train_steps} training steps) on '{args.condition}', {n} episodes")
print(f"  crash rate   : {k / n:.0%}   (95% CI {lo:.0%} to {hi:.0%})")
print(f"  return       : {df['return'].mean():.2f} ± {df['return'].std():.2f}")
print(f"  distance     : {df['distance_m'].mean():.0f} m on average")
print(f"  mean speed   : {df['mean_speed_mps'].mean():.1f} m/s")
print(f"Appended {n} rows to {args.out}")