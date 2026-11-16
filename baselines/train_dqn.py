"""
Train the DQN baseline on the "normal" condition.

Run from the repo's top folder:
    python -m baselines.train_dqn --seed 0
Saves models/dqn_seed0.zip and TensorBoard logs in logs/dqn/.
"""
import argparse
import os
import time

from stable_baselines3 import DQN

from envs.conditions import make_env

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="training seed (use 0, 1, 2, ...)")
parser.add_argument("--steps", type=int, default=20_000, help="total training steps")
args = parser.parse_args()

env = make_env("normal")

model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=15_000,
    learning_starts=200,
    batch_size=32,
    gamma=0.8,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=50,
    exploration_fraction=0.7,
    seed=args.seed,
    verbose=1,
    tensorboard_log="logs/dqn",
    device="cpu",   # a tiny network: the CPU is as fast as the GPU here
)

start = time.time()
model.learn(total_timesteps=args.steps, tb_log_name=f"seed{args.seed}", progress_bar=True)
print(f"\nTraining took {(time.time() - start) / 60:.1f} minutes")

os.makedirs("models", exist_ok=True)
path = f"models/dqn_seed{args.seed}"
model.save(path)
print(f"Saved {path}.zip")
