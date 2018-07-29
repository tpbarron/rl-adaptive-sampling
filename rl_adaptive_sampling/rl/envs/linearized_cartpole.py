import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy
# from os import path
# import control
import pygame

def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.

    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    return K, X, eigVals

class LinearizedCartPole(gym.Env):

    def __init__(self):

        self.viewer = None

        self.dt = 0.01 #1.0/60.0
        self.nu = 13.2
        self.g = 9.8

        self.mu0 = 0.0
        self.var0 = 0.1
        self.vart = 0.01 * self.var0

        self.A =  np.array([[1.0, self.dt, 0.0,               0.0],
                            [0.0, 1.0,     0.0,               0.0],
                            [0.0, 0.0,     1.0,               self.dt],
                            [0.0, 0.0,     self.nu * self.dt, 1.0]])
        self.B = np.zeros((4, 1))
        self.B[1] = self.dt
        self.B[3] = self.nu * self.dt / self.g

        self.d, self.p = self.B.shape

        self.q = np.array([1.25, 1.0, 12.0, 0.25])
        self.Q = np.eye(self.d) * self.q
        print (self.Q)

        self.r = 0.01
        self.R = np.eye(self.p) * self.r

        self.K,S,E = dlqr(self.A, self.B, self.Q, self.R)
        self.K = np.asarray(self.K)
        # print (self.K)
        # input("")

        self.T = 100
        self.time = 0

        self.action_space = spaces.Box(low=-1e+8, high=1e+8, shape=(self.p,))
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.d, ))

        self._seed()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, u, use_k=False):
        # print (self.K.shape, self.state.shape)
        if use_k:
            u = -np.dot(self.K, self.state)
        x = self.state
        cost = np.dot(x, np.dot(self.Q, x)) + np.dot(u, np.dot(self.R, u))
        new_x = np.dot(self.A, x) + np.dot(self.B, u) + self.np_random.normal(0, self.vart, size=self.d)
        self.state = new_x

        terminated = False
        if np.abs(x[2]) >= np.pi/6.0 or self.time > self.T:
            terminated = True

        self.time += 1

        return self._get_obs(), cost, terminated, {}


    def reset(self):
        self.state = self.np_random.normal(self.mu0, self.var0, size = self.d)
        self.time = 0
        return self._get_obs()


    def get_pixels_from_state(self):
        x,y = self.state[0:2]
        bounds_x = (-1.0, 1.0)
        bounds_y = (-1.0, 1.0)

        px = (x - bounds_x[0]) / (bounds_x[1] - bounds_x[0]) * self.screen_w
        py = (y - bounds_y[0]) / (bounds_y[1] - bounds_y[0]) * self.screen_h

        # print (px, py)
        return int(round(px)), int(round(py))

    def render(self):
        return None

    def _get_obs(self):
        return  self.state

    def get_params(self):
        return self.A, self.B, self.Q, self.R
