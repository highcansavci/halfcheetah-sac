import pybullet_data
pybullet_data.getDataPath = lambda: r"C:\pybullet_data"

import gymnasium as gym
import pybullet_envs_gymnasium

env = gym.make("HalfCheetahBulletEnv-v0")
obs, _ = env.reset()

heights = []
for _ in range(500):
    obs, _, term, trunc, _ = env.step(env.action_space.sample())
    h = env.unwrapped.robot.body_xyz[2]
    heights.append(h)
    if term or trunc:
        obs, _ = env.reset()

env.close()

import numpy as np
print(f"min:    {min(heights):.3f}")
print(f"mean:   {np.mean(heights):.3f}")
print(f"max:    {max(heights):.3f}")
print(f"p10:    {np.percentile(heights, 10):.3f}")
print(f"p25:    {np.percentile(heights, 25):.3f}")