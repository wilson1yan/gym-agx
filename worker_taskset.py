import sys
import time
import random
from tqdm import tqdm
import numpy as np

import gym
from gym_agx import envs
from gym_agx.utils.agx_utils import to_numpy_array

def compute_action(env, idx):
    pusher_pos = to_numpy_array(env.pusher.getRigidBody('pusher').getPosition())[:2]
    segment_pos = to_numpy_array(env.rope.segments[idx].getRigidBody().getPosition())[:2]
    direction = segment_pos - pusher_pos
    direction /= np.linalg.norm(direction)
    return direction


def run(i, env_name):
    env = gym.make(env_name, reward_type='sparse', observation_type='rgb', headless=0)
    n_segs = len(env.rope.segments)
    idx = random.randint(int(0.1 * n_segs), int(0.9 * n_segs))
    policy = lambda: compute_action(env, idx)

    pbar = tqdm(total=10, position=i)
    start = time.time()
    frames = 0
    for _ in range(10):
        env.reset()
        idx = random.randint(int(0.1 * n_segs), int(0.9 * n_segs))
        done = False
        while not done:
            obs, reward, done, _ = env.step(policy()))
            frames += 1
            fps = frames / (time.time() - start)
            pbar.set_description(f"FPS: {fps}")
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    i, env_name = sys.argv[1], sys.argv[2]
    i = int(i)
    run(i, env_name)