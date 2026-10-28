"""
Step 3: Train a DQN agent on highway-fast-v0 (settings from the highway-env quickstart).
Run:  python train_dqn.py
Saves the trained agent as dqn_highway.zip and training logs in logs/.
"""
import time
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN

env = gym.make("highway-fast-v0")   # a faster version of the highway environment

model = DQN(
    "MlpPolicy",                    # a plain fully-connected network (input = 25 numbers)
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=15_000,             # how many past steps are remembered
    learning_starts=200,            # random driving first, then start learning
    batch_size=32,
    gamma=0.8,                      # how much the agent cares about future reward
    train_freq=1,
    gradient_steps=1,
    target_update_interval=50,
    exploration_fraction=0.7,       # explores randomly a lot at first, less later
    verbose=1,
    tensorboard_log="logs/",
    device="auto",                  # uses GPU if available (CPU is also fine here)
)

start = time.time()
model.learn(total_timesteps=20_000, progress_bar=True)
print(f"\nTraining took {(time.time() - start) / 60:.1f} minutes")

model.save("dqn_highway")
print("Saved trained agent to dqn_highway.zip")
