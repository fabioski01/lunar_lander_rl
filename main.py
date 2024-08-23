# import environment
import gymnasium as gym
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.functional import mse_loss

from torch import optim
import copy
from collections import namedtuple

from itertools import count
import math

import matplotlib.pyplot as plt
%matplotlib inline

# Deep Q Learning Network
# Using a deep Q network to solve the discrete lunar lander challenge. https://gym.openai.com/envs/LunarLander-v2/
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for
# moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from
# landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional
# -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200
# points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on
# its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine,
# fire right orientation engine.

# create the environment and explore the action and observation space.
env = gym.make('LunarLander-v2')
# env = gym.make(
#     "LunarLander-v2",
#     continuous: bool = False,
#     gravity: float = -10.0,
#     enable_wind: bool = False,
#     wind_power: float = 15.0,
#     turbulence_power: float = 1.5,
# )

print('Example action {}'.format(env.action_space.sample()))
print('Example observation space {}'.format(env.reset()))
