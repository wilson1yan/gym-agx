import numpy as np
import time
import gym
import copy
import random

from gym_agx import envs
from gym_agx.utils.agx_utils import to_numpy_array


def compute_action(idx):
    pusher_pos = to_numpy_array(env.pusher.getRigidBody('pusher').getPosition())[:2]
    segment_pos = np.array([to_numpy_array(seg.getRigidBody().getPosition())
                            for seg in env.rope.segments])[:, :2]
    direction = segment_pos[idx] - pusher_pos
    direction /= np.linalg.norm(direction)

    line_points = np.linspace(0, 1, 10)[:, None] * (segment_pos[idx] - pusher_pos)[None] + pusher_pos[None]

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(segment_pos[:, 0], segment_pos[:, 1])
    # plt.scatter([pusher_pos[0]], [pusher_pos[1]])
    # plt.scatter(line_points[:, 0], line_points[:, 1])
    # plt.show()

    return direction

env = gym.make("RopeObstacle-v2", reward_type="dense", observation_type="rgb", headless=0)
observation = env.reset()
n_segs = len(env.rope.segments)
idx = random.randint(int(0.1 * n_segs), int(0.9 * n_segs))

for i in range(1000):
    env.render("osg")
    action = compute_action(idx)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
        idx = random.randint(int(0.4 * n_segs), int(0.6 * n_segs))
env.close()
