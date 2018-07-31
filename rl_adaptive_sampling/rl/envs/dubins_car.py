
import gym
from gym import spaces
import numpy as np

class DubinsCar(gym.Env):

    def __init__(self):
        super(DubinsCar, self).__init__()
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.L = 1.0
        self.dt = 0.01
        self.horizon = 100

        self.t = 0
        self.var0 = 0.0001
        self.state = np.array([self.x, self.y, self.theta])

        self.goal = np.array([1, 1, 0.0])

        self.action_space = spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        self.observation_space = spaces.Box(np.array([-np.inf, -np.inf, -np.pi]), np.array([-np.inf, np.inf, np.pi]))

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0]) #+ np.ones_like(self.state) * self.var0
        self.t = 0
        return self.state

    def step(self, u):
        assert self.action_space.contains(u)
        # print ("Control: ", u)
        self.t += 1

        u_s = u[0]
        u_theta = u[1] * np.pi / 4.0 # a steering range of 2 pi / 4.

        x_dot = u_s * np.cos(self.theta)
        y_dot = u_s * np.sin(self.theta)
        theta_dot = u_s / self.L * np.tan(u_theta)

        self.x = self.x + x_dot * self.dt
        self.y = self.y + y_dot * self.dt
        self.theta = self.theta + theta_dot * self.dt

        # clamp theta
        if self.theta < -np.pi:
            self.theta = 2 * np.pi + self.theta
        elif self.theta > np.pi:
            self.theta = 2 * np.pi - self.theta

        self.state = np.array([self.x, self.y, self.theta])
        rew = -np.linalg.norm(self.state[0:2] - self.goal[0:2])

        done = False
        if self.t > self.horizon:
            done = True
        if np.linalg.norm(self.state - self.goal) < 0.5:
            done = True

        # print ("State: ", self.state)
        # input("")

        return self.state, rew, done, {}
