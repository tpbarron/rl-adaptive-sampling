

import gym
from gym import spaces

class PointMass(gym.Env):

    def __init__(self):
        self.A = None
        self.B = None
        self.x = None
        self.dt = 0.05

    def reset(self):
        # set random initial x
        pass

    def step(self, u):
        # compute xt+1 and cost
        pass
