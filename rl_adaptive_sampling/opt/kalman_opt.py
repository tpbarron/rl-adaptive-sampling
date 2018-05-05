import numpy as np
from scipy import linalg

class KalmanFilter(object):

    def __init__(self,
                 state_dim,
                 error_threshold=0.01,
                 use_diagonal_approx=True):
        self.state_dim = state_dim
        self.error_threshold = error_threshold
        self.use_diagonal_approx = use_diagonal_approx

        self.ndim = state_dim
        self.Rt = None
        self.xt = None
        self.y = None
        self.xt_old = None
        self.reset()

    def reset(self, sos_init=0, err_init=1.0, reset_observation_noise=False):
        if self.xt is None:
            self.xt = np.zeros((self.state_dim,1)) #self.flatten_params()
        if self.y is None:
            self.y = np.zeros((self.state_dim,1))
        self.ndim = self.xt.size

        if self.use_diagonal_approx:
            self.Pt = np.ones((self.ndim,1)) * err_init ** 2.0
            if reset_observation_noise or self.Rt is None:
                self.Rt = np.ones((self.ndim,1)) * 10
            self.ones = np.ones((self.ndim,1))
            self.e = np.zeros((self.ndim,1))
        else:
            self.Pt = np.eye(self.ndim) * err_init ** 2.0
            if reset_observation_noise or self.Rt is None:
                self.Rt = np.eye(self.ndim) * 10
            self.I = np.eye(self.ndim)
            self.e = np.zeros((self.ndim, 1))

        self.I = np.eye(self.ndim)

        self.mean = np.zeros((self.ndim, 1))
        self.var = np.zeros((self.ndim, 1))
        # self.sos = np.zeros((self.ndim,1))
        # self.sos.fill(sos_init) # initializing this high, gives conservative init
        # print ("OLD: ", self.xt_old)
        self.e.fill(err_init)
        self.n = 0
        self.do_step = False
        # self.xt_old = np.copy(self.xt)

    def update(self, grad):
        """
        Optimized update
        """
        y = grad
        self.y = y
        self.n += 1
        mean_past = self.mean
        self.mean = self.mean + (y - self.mean) / self.n
        if self.n > 1:
            self.var = ((self.n-1) * self.var + (y-mean_past) * (y - self.mean))/self.n
            self.Rt = self.var #np.diag(self.var[:,0])

        # Kt = self.Pt @ np.linalg.pinv(self.Pt + self.Rt)
        # Et = y - self.xt
        # self.Pt = (np.eye(self.ndim) - Kt) @ self.Pt
        # self.xt = self.xt + Kt @ Et
        # self.e = (np.eye(self.ndim) - Kt) @ self.e

        # print ("shapes: ", self.Pt.shape, Kt.shape, Et.shape, self.xt.shape, self.e.shape)
        # # update running mean / var, do this before computing K!
        # self.n += 1
        # mean_past = self.mean
        # self.mean = self.mean + (y - self.mean) / self.n
        # if self.use_diagonal_approx:
        #     # self.sos = self.sos + (y - mean_tmp) * (y - self.mean)
        #     # var = self.sos / self.n
        #     self.var = ((self.n-1) * self.var + (y-mean_past) * (y - self.mean))/self.n
        #     # Rt = np.diag(var)
        #     if self.n > 1:
        #         # self.Rt = self.var # leave as vector, makes for easier inversion of diag matrix
        #         self.Rt = np.diag(self.var)
        #
        #     # print (self.Rt.shape)
        #     # input("")
        # else:
        #     x = (y - self.mean)[np.newaxis,:] # now 1 x N
        #     delta = x @ np.transpose(x) * (self.n) / (self.n+1)
        #     if self.n > 1:
        #         self.Rt = self.n * self.Rt + delta
        #

        Et = y - self.xt

        # NOTE: this is being computed properly but was being overwhelmed by magnitude of Pt
        if self.use_diagonal_approx:
            Kt = self.Pt * 1.0/(self.Pt + self.Rt)
            self.Pt = (self.ones - Kt) * self.Pt
            self.xt = self.xt + Kt * Et
            self.e = (self.ones - Kt) * self.e
        else:
            Kt = self.Pt @ np.linalg.pinv(self.Pt + self.Rt)
            self.Pt = (self.I - Kt) @ self.Pt
            self.xt = self.xt + Kt @ Et
            self.e = (self.I - Kt) @ self.e
