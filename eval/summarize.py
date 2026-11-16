"""
Summarise results/eval.csv: one line per training run, then one line per agent across seeds.

Run from the repo's top folder:
    python -m eval.summarize
    python -m eval.summarize --condition normal --csv results/eval.csv

Prints Markdown tables you can paste into the README, and saves results/summary.csv.
"""
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="results/eval.csv")
parser.add_argument("--condition", default="normal")
args = parser.parse_args()

df = pd.read_csv(args.csv)
df = df[df["condition"] == args.condition]

# 1) one row per training run (agent + seed): the unit we average over
runs = (df.groupby(["agent", "train_seed", "train_steps"])
          .agg(episodes=("crashed", "size"),
               crash_rate=("crashed", "mean"),
               ret=("return", "mean"),
               distance=("distance_m", "mean"),
               speed=("mean_speed_mps", "mean"))
          .reset_index())

# 2) one row per agent: mean across seeds, plus the spread between seeds
per_agent = (runs.groupby("agent")
               .agg(seeds=("train_seed", "nunique"),
                    crash_mean=("crash_rate", "mean"),
                    crash_min=("crash_rate", "min"),
                    crash_max=("crash_rate", "max"),
                    crash_std=("crash_rate", "std"),
                    ret_mean=("ret", "mean"),
                    ret_std=("ret", "std"),
                    distance=("distance", "mean"),
                    speed=("speed", "mean"))
               .reset_index())


def pct(x):
    return f"{100 * x:.0f}%"


def sd(x, scale=1.0, fmt="{:.1f}"):
    return "–" if pd.isna(x) else fmt.format(scale * x)


print(f"\n### Per training run — condition '{args.condition}'\n")
print("| Agent | Seed | Train steps | Episodes | Crash rate | Return | Distance (m) | Speed (m/s) |")
print("|---|---|---|---|---|---|---|---|")
for r in runs.itertuples():
    print(f"| {r.agent} | {r.train_seed} | {r.train_steps} | {r.episodes} | {pct(r.crash_rate)} "
          f"| {r.ret:.2f} | {r.distance:.0f} | {r.speed:.1f} |")

print(f"\n### Per agent, across seeds — condition '{args.condition}'\n")
print("| Agent | Seeds | Crash rate: mean ± std (min–max) | Return: mean ± std | Distance (m) | Speed (m/s) |")
print("|---|---|---|---|---|---|")
for a in per_agent.itertuples():
    print(f"| {a.agent} | {a.seeds} "
          f"| {pct(a.crash_mean)} ± {sd(a.crash_std, 100, '{:.0f}')} ({pct(a.crash_min)}–{pct(a.crash_max)}) "
          f"| {a.ret_mean:.2f} ± {sd(a.ret_std, 1, '{:.2f}')} | {a.distance:.0f} | {a.speed:.1f} |")

per_agent.to_csv("results/summary.csv", index=False)
print("\nSaved results/summary.csv")