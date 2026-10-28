"""
Step 2: Check that everything is installed and watch a RANDOM agent drive.
Run:  python check_setup.py
A window should open showing the car. Close it or wait for 3 episodes to finish.
"""
import torch
import gymnasium as gym
import highway_env  # noqa: F401  (importing this registers the highway environments)

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

env = gym.make("highway-v0", render_mode="human")

for ep in range(3):
    obs, info = env.reset(seed=ep)
    done = truncated = False
    total_reward, steps = 0.0, 0
    while not (done or truncated):
        action = env.action_space.sample()          # random choice out of the 5 actions
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    print(f"episode {ep}: decisions={steps}, return={total_reward:.2f}, crashed={info['crashed']}")

env.close()
print("observation shape:", obs.shape, "(rows = ego car + 4 nearby cars, columns = features)")
print("actions:", env.unwrapped.action_type.actions)
