import numpy as np
import time
import gym
import copy
import random
from tqdm import tqdm

from gym_agx import envs
from gym_agx.utils.agx_utils import to_numpy_array


def compute_action(idx):
    pusher_pos = to_numpy_array(env.pusher.getRigidBody('pusher').getPosition())[:2]
    segment_pos = to_numpy_array(env.rope.segments[idx].getRigidBody().getPosition())[:2]
    direction = segment_pos - pusher_pos
    direction /= np.linalg.norm(direction)
    return direction

env = gym.make("RopeObstacle-v2", reward_type="sparse", observation_type="rgb", headless=0)
observation = env.reset()
n_segs = len(env.rope.segments)
idx = random.randint(int(0.1 * n_segs), int(0.9 * n_segs))

start = time.time()
pbar = tqdm(total=10000)
for i in range(10000):
    action = compute_action(idx)
    observation, reward, done, info = env.step(action)
    pbar.set_description(f"{(i + 1) / (time.time() - start)} FPS")
    pbar.update(1)

    if done:
        observation = env.reset()
        idx = random.randint(int(0.4 * n_segs), int(0.6 * n_segs))
pbar.close()
env.close()
