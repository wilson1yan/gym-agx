# Gym AGX

This project contains a set of deformable object manipulation environments, using AGX Dynamics as the simulation software. For the moment, gym-agx consists of deformable linear objects (DLOs) only, e.g. ropes, cables. We categorize two types of manipulation in the context of de deformable objects:

1) Explicit shape control - the goal is to deform an object to a specific shape
2) Implicit shape control - the goal is to move the object into a specific configuration, indirectly needing to be deformed

## Explicit shape control

![BendWire](https://drive.google.com/file/d/1oa98fspwVYnNulq5SEgSYkrrTLXTz-_Y/view?usp=sharing)
![BendWireObstacle](https://drive.google.com/file/d/16qEuWRvFr9B5u46lRJM77_tCy22VEoNz/view?usp=sharing)
![PushRope](https://drive.google.com/file/d/1IuuDTLa-7hNP373yuFJfEfU3hZt9dG5Q/view?usp=sharing)

## Implicit shape control

![PegInHole](https://drive.google.com/file/d/1OT4GVt3xCTo2mkpzOVuaASToH36y2Jde/view?usp=sharing)
![CableClosing](https://drive.google.com/file/d/12pEKXavw5ox01l7-NK6JoqVKPVU8ArHz/view?usp=sharing)
![RubberBand](https://drive.google.com/file/d/1halbLByFBRG6yQ0wG-icG46lCcOL3Pw4/view?usp=sharing)

## Getting Started

To install the environment, just go to the folder containing the gym-agx folder and run the command:

```
pip install -e gym-agx
```

This will install the gym environment. Now, you can use your gym environment with the following:

```
import gym
from gym_agx import envs

env = gym.make('BendWire-v0')
```
