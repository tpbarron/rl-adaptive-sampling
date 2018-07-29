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

class LQG_Env(gym.Env):

    def __init__(self, state0=np.array([0.5, 0.5, 0.0, 0.0])):

        self.viewer = None

        self.dt = 0.05
        self.m = 1

        self.A =  np.array([[1.0, 0.0, self.dt, 0.0],
                            [0.0, 1.0, 0.0, self.dt],
                            [0., 0.0, 1.0, 0.0],
                            [0., 0.0, 0.0, 1.0]])
        self.B = np.zeros((4, 2))
        self.B[2,0] = self.dt / self.m
        self.B[3,1] = self.dt / self.m

        self.d, self.p = self.B.shape

        self.q = 1.0
        self.r = 0.01
        self.Q = np.eye(self.d) * self.q
        self.R = np.eye(self.p) * self.r

        self.K,S,E = dlqr(self.A, self.B, self.Q, self.R)
        self.K = np.asarray(self.K)
        # print (self.K)
        # input("")
        self.mu0 = state0
        print ("mu0", self.mu0)
        # self.mu0 = np.array([0.5, 0.5, 0.25, 0.5])
        self.var0 = 0.0001

        self.time = 0
        self.T = 100

        self.action_space = spaces.Box(low=-1e+8, high=1e+8, shape=(self.p,))
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.d, ))

        pygame.init()
        self.screen_w = 200
        self.screen_h = 200
        self.screen = None

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
        new_x = np.dot(self.A, x) + np.dot(self.B, u) + self.np_random.normal(0, self.var0, size=self.d)
        self.state = new_x

        terminated = False
        self.time += 1
        if self.time >= self.T:
            terminated = True

        return self._get_obs(), cost, terminated, {}

    def reset(self):
        self.state = self.np_random.normal(self.mu0, self.var0, size = self.d)
        self.last_u = None
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
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption('LQR')

        # Fill background
        background = pygame.Surface(self.screen.get_size())
        background = background.convert()
        background.fill((250, 250, 250))
        self.screen.blit(background, (0, 0))

        px, py = self.get_pixels_from_state()
        pygame.draw.circle(self.screen, (255,0,0), (px, py), 2)

        pygame.display.flip()
        pygame.display.update()


    def _get_obs(self):
        return  self.state

    def get_params(self):
        return self.A, self.B, self.Q, self.R
