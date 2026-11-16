"""
Compare how fast the two environments simulate, with a random agent.
Run from the repo's top folder:
    python scripts/speed_test.py
"""
import time
import gymnasium as gym
import highway_env  # noqa: F401

N_STEPS = 1000

for env_id in ["highway-fast-v0", "highway-v0"]:
    env = gym.make(env_id)
    env.reset(seed=0)
    episodes = 1
    start = time.time()
    for _ in range(N_STEPS):
        _, _, done, truncated, _ = env.step(env.action_space.sample())
        if done or truncated:
            env.reset()
            episodes += 1
    secs = time.time() - start
    cfg = env.unwrapped.config
    print(f"{env_id:16s}: {N_STEPS / secs:6.1f} steps/sec  ({secs:.1f} s for {N_STEPS} steps, {episodes} episodes)")
    print(f"{'':18s}lanes={cfg['lanes_count']}, cars={cfg['vehicles_count']}, "
          f"physics Hz={cfg['simulation_frequency']}, episode={cfg['duration']} s")
    env.close()