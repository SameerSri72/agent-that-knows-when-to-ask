"""
One place that creates every environment used in the project.

Training happens on "normal" only. The unseen test conditions
(dense traffic, aggressive drivers, fewer lanes, merge) are added
here in Week 4, so every script picks them up automatically.
"""
import gymnasium as gym
import highway_env  # noqa: F401  (importing registers the highway environments)

BASE_ENV = "highway-fast-v0"   # fast variant: fewer cars, coarser physics, trains much faster

# condition name -> settings that override the base environment's defaults
CONDITIONS = {
    "normal": {},
}


def make_env(condition="normal", render_mode=None):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition '{condition}'. Choose from: {list(CONDITIONS)}")
    return gym.make(BASE_ENV, render_mode=render_mode, config=CONDITIONS[condition])
