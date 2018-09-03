
import gym
from gym import spaces
import numpy as np
from scipy.spatial.distance import cityblock

class DubinsCar(gym.Env):

    def __init__(self):
        super(DubinsCar, self).__init__()
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.L = 1.0
        self.dt = 0.1
        self.horizon = 100

        self.t = 0
        self.var0 = 0.0001
        # x, y, xdot, ydot, theta, gx, gy
        self.state = np.array([self.x, self.y, 0.0, 0.0, self.theta, 0.0, 0.0])

        self.goal = np.array([25.0, -5.0, 0.0])

        self.action_space = spaces.Box(np.array([0.0, -1.0]), np.array([1.0, 1.0]))
        self.observation_space = spaces.Box(np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.inf, -np.inf]), np.array([np.inf, np.inf, np.inf, np.inf, np.pi, np.inf, np.inf]))

    def reset(self):
        # The state is x,y, xdot, ydot, theta
        self.state = np.random.normal(np.zeros((5,)), np.ones((5,)) * self.var0)
        self.goal = np.array([25.0, np.random.randint(-2, 2), 0.0])
        unit_vec_to_goal = self.goal[0:2]-self.state[0:2] / np.linalg.norm(self.goal[0:2]-self.state[0:2])
        self.state = np.concatenate((self.state, unit_vec_to_goal))
        self.x, self.y = 0.0, 0.0
        self.t = 0.0
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

        prev_xy = np.array([self.x, self.y])

        self.x = self.x + x_dot * self.dt
        self.y = self.y + y_dot * self.dt
        self.theta = self.theta + theta_dot * self.dt

        # clamp theta
        if self.theta < -np.pi:
            self.theta = 2 * np.pi + self.theta
        elif self.theta > np.pi:
            self.theta = 2 * np.pi - self.theta

        unit_vec_to_goal = self.goal[0:2]-self.state[0:2] / np.linalg.norm(self.goal[0:2]-self.state[0:2])
        self.state = np.concatenate((np.array([self.x, self.y, x_dot, y_dot, self.theta]) + np.random.normal(0, self.var0, size=5), unit_vec_to_goal))

        prev_dist = np.linalg.norm(prev_xy - self.goal[0:2])
        current_dist = np.linalg.norm(self.state[0:2] - self.goal[0:2])
        rew = 0.0
        if current_dist < prev_dist:
            rew = 1.0
        else:
            rew = -1.0
        #rew = -np.linalg.norm(self.state[0:2] - self.goal[0:2])

        done = False
        if self.t >= self.horizon:
            done = True
        if np.linalg.norm(self.state[0:2] - self.goal[0:2]) < 0.5:
            done = True
        # if done:
        #     print ("Distances: ", u, prev_xy, self.state[0:2], prev_dist, current_dist)
        #     input("")

        # print ("State: ", self.state)
        # input("")

        return self.state, rew, done, {}
