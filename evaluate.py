"""
Step 4: Compare the RANDOM agent with the TRAINED agent and save GIFs of both.
Run:  python evaluate.py
Prints crash rate and average return, saves random.gif and trained.gif.
"""
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import imageio
from stable_baselines3 import DQN

N_EPISODES = 30


def run(policy, n=N_EPISODES, gif_path=None):
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    crashes, returns, frames = 0, [], []
    for ep in range(n):
        obs, info = env.reset(seed=1000 + ep)      # same test roads for both agents
        done = truncated = False
        total = 0.0
        while not (done or truncated):
            action = policy(obs, env)
            obs, reward, done, truncated, info = env.step(action)
            total += reward
            if gif_path and ep == 0:
                frames.append(env.render())
        crashes += int(info["crashed"])
        returns.append(total)
    env.close()
    if gif_path:
        imageio.mimsave(gif_path, frames, duration=0.3)   # one frame per decision
    return crashes / n, float(np.mean(returns)), float(np.std(returns))


model = DQN.load("dqn_highway")

random_policy = lambda obs, env: env.action_space.sample()
trained_policy = lambda obs, env: int(model.predict(obs, deterministic=True)[0])

print(f"Evaluating on {N_EPISODES} episodes each...\n")
r_crash, r_mean, r_std = run(random_policy, gif_path="random.gif")
t_crash, t_mean, t_std = run(trained_policy, gif_path="trained.gif")

print(f"{'Agent':<10}{'Crash rate':>12}{'Avg return':>14}")
print(f"{'Random':<10}{r_crash:>12.0%}{r_mean:>10.2f} ± {r_std:.2f}")
print(f"{'Trained':<10}{t_crash:>12.0%}{t_mean:>10.2f} ± {t_std:.2f}")
print("\nSaved random.gif and trained.gif — open them to watch both agents.")
