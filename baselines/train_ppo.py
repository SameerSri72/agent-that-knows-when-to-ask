"""
Train the PPO baseline on the "normal" condition.

Run from the repo's top folder:
    python -m baselines.train_ppo --seed 0
Saves models/ppo_seed0.zip and TensorBoard logs in logs/ppo/.

PPO collects fresh experience from 4 copies of the environment at once,
updates the policy, throws that experience away, and repeats (guide, Part 5.4).
"""
import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.conditions import make_env

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="training seed (use 0, 1, 2, ...)")
parser.add_argument("--steps", type=int, default=50_000, help="total training steps (all 4 envs together)")
args = parser.parse_args()

env = make_vec_env(lambda: make_env("normal"), n_envs=4, seed=args.seed)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),  # actor and critic networks
    n_steps=256,        # steps per env before each update -> 4 x 256 = 1024 steps per update
    batch_size=64,
    n_epochs=10,        # passes over each batch of fresh experience
    learning_rate=5e-4,
    gamma=0.8,          # same horizon as DQN, for a fair comparison
    clip_range=0.2,     # the PPO "clip": ratio kept within 0.8 to 1.2
    seed=args.seed,
    verbose=1,
    tensorboard_log="logs/ppo",
    device="cpu",
)

start = time.time()
model.learn(total_timesteps=args.steps, tb_log_name=f"seed{args.seed}", progress_bar=True)
print(f"\nTraining took {(time.time() - start) / 60:.1f} minutes")

os.makedirs("models", exist_ok=True)
path = f"models/ppo_seed{args.seed}"
model.save(path)
print(f"Saved {path}.zip")
