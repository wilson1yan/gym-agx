import numpy as np
import time
import gym
import copy
import random
from tqdm import tqdm

from gym_agx import envs
from gym_agx.utils.agx_utils import to_numpy_array

env = gym.make("RopeObstacle-v2", reward_type="sparse", observation_type="rgb", headless=0)
observation = env.reset()
policy = env.construct_policy()

start = time.time()
pbar = tqdm(total=10000)
for i in range(10000):
    observation, reward, done, info = env.step(policy())
    pbar.set_description(f"{(i + 1) / (time.time() - start)} FPS")
    pbar.update(1)

    if done:
        observation = env.reset()
        policy = env.construct_policy()
pbar.close()
env.close()
