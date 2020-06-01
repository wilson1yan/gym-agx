import math
import logging
import gym_wrappers
from gym_agx.utils.utils import sinusoidal_trajectory

N_EPISODES = 1
EPISODE_LENGTH = 3000

logger = logging.getLogger('gym_agx.tests')


def main():
    env = gym_wrappers.make('PushRopeDense-v0')
    for _ in range(N_EPISODES):
        env.reset()
        for t in range(EPISODE_LENGTH):
            env.render()
            obs, reward, done, info = env.step(env.action_space.sample())
            print(reward)
    env.close()


if __name__ == "__main__":
    main()
