import numpy as np
from scipy import linalg

class KalmanFilter(object):

    def __init__(self,
                 state_dim,
                 use_last_error=False,
                 min_error_init=0.01,
                 use_diagonal_approx=True,
                 sos_init=10.0,
                 error_init=1.0,
                 reset_observation_noise=False,
                 reset_state=False):
        self.state_dim = state_dim
        self.use_last_error = use_last_error
        self.min_error_init = min_error_init
        self.use_diagonal_approx = use_diagonal_approx
        self.sos_init = sos_init
        self.error_init = error_init
        self.reset_observation_noise = reset_observation_noise
        self.reset_state = reset_state

        attrs = vars(self)
        print ('KF parameters:' + ', '.join("%s: %s" % item for item in attrs.items()))

        self.Pt = None
        self.Rt = None
        self.mean = None
        self.xt = None
        self.xt_old = None
        self.e = None

        if self.use_diagonal_approx:
            self.ones = np.ones((self.state_dim, 1))
        else:
            self.I = np.eye(self.state_dim)

        # total steps; used for variance and running mean
        self.steps = 0

        # windowed mean / var
        self.window_size = 100
        self.window_index = 0
        self.window_buffer = np.empty((self.window_size, self.state_dim, 1))

    def reset(self):
        # set expected error
        if self.use_last_error and self.xt is not None:
            self.e = (self.xt - self.xt_old)
            # if less than threshold, set to minimum preserving sign
            # print ("ERROR: ", self.e[np.abs(self.e) < self.min_error_init])
            # self.e[np.abs(self.e) < self.min_error_init] = np.sign(self.e[np.abs(self.e) < self.min_error_init]) * self.min_error_init
        else:
            # either xt is None or use_last_error = False
            if self.e is None:
                self.e = np.zeros((self.state_dim, 1))
            self.e.fill(self.error_init)

        if self.use_diagonal_approx:
            if self.reset_observation_noise or self.Rt is None:
                self.Rt = np.ones((self.state_dim, 1)) * 10
            expected_error = self.e**2.0 + np.ones((self.state_dim, 1)) * 1e-8 if self.use_last_error and self.Pt is not None else np.ones((self.state_dim, 1)) * self.error_init
        else:
            if self.reset_observation_noise or self.Rt is None:
                self.Rt = np.eye(self.state_dim) * 10
            expected_error = np.dot(self.e, np.transpose(self.e)) + np.eye(self.state_dim) * 1e-8 if self.use_last_error and self.Pt is not None else np.eye(self.state_dim) * self.error_init

        self.Pt = expected_error
        if self.reset_observation_noise or self.mean is None or self.sos is None:
            self.mean = np.zeros((self.state_dim, 1))
            self.sos = np.zeros((self.state_dim, 1))
            # initializing this high, gives conservative init
            self.sos.fill(self.sos_init)
            self.steps = 0

        # set xt on first step
        if self.reset_state or self.xt is None:
            self.xt = np.zeros((self.state_dim, 1))

        # print ("OLD: ", self.xt_old)
        self.n = 0
        self.do_step = False
        self.xt_old = np.copy(self.xt)
        # self.xt = np.zeros((self.state_dim, 1))

    def compute_running_cov(self, y):
        """
        See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance
        """
        # update mean; do this first to try to get better cov est
        mean_past = self.mean
        self.mean = self.mean + (y - self.mean) / self.n
        # assume n has already been incremented
        if self.n > 1: # don't compute cov when only one sample
            resid1 = y - mean_past # note this is the mean BEFORE seeing the sample y!
            resid2 = y - self.mean
            Y = np.dot(resid2, np.transpose(resid1)) # outer product
            self.Rt = ((self.n - 1) * self.Rt) / self.n + Y / self.n

    def update(self, y):
        """
        Optimized update
        """
        # print ("Kalman update")
        # always increment n first!
        self.n += 1
        self.steps += 1

        if self.use_diagonal_approx:
            mean_past = self.mean.copy()
            if self.steps <= self.window_size:
                # print ("Not enough samples for window yet")
                # no subtraction necessary
                self.mean = self.mean + (y - self.mean) / self.steps
                self.sos = self.sos + (y - mean_past) * (y - self.mean)
                # add to buffer
                self.window_buffer[self.window_index] = y.copy()
                # print (self.steps, self.window_index)
            else:
                # print ("Using windowed update")
                # remove old sample
                # https://stackoverflow.com/questions/5147378/rolling-variance-algorithm
                # new_mean = mean + (x_new - xs[next_index])/window_size;
                # varSum = var_sum + (x_new - mean) * (x_new - new_mean) - (xs[next_index] - mean) * (xs[next_index] - new_mean);

                last_index = (self.window_index+1) % self.window_size
                yold = self.window_buffer[last_index]
                self.mean  = self.mean + y / self.window_size - yold / self.window_size
                self.sos = self.sos + (y + yold - mean_past - self.mean) * (y - yold)
                # self.sos = self.sos + (y - mean_past) * (y - self.mean) - (yold - mean_past) * (yold - self.mean)
                # overwrite the oldest element
                self.window_buffer[last_index] = y

            self.window_index += 1
            self.window_index %= self.window_size

            #print ("m, sos: ", self.mean.shape, self.sos.shape)
            if self.steps > 1:
                if self.steps <= self.window_size:
                    var = self.sos / self.steps #(self.steps-1)
                    # print ("No window var: ", var, self.mean)
                else:
                    var = self.sos / self.steps #(self.window_size-1)
                    # print ("Window var: ", var, self.mean)
                # print ("var: ", var)
                # input("")
                self.Rt = var # leave as vector, makes for easier inversion of diag matrix
        else:
            # import time
            # start = time.time()
            # self.compute_running_cov(y)
            # end = time.time()
            # print ("Time for cov update: ", (e-s))

            mean_past = self.mean.copy()
            if self.steps <= self.window_size:
                # print ("Not enough samples for window yet")
                # no subtraction necessary
                self.mean = self.mean + (y - self.mean) / self.steps
                self.sos = self.sos + np.dot((y - mean_past), np.transpose((y - self.mean)))
                # add to buffer
                self.window_buffer[self.window_index] = y.copy()
                # print (self.steps, self.window_index)
            else:
                # print ("Using windowed update")
                # remove old sample
                # https://stackoverflow.com/questions/5147378/rolling-variance-algorithm
                # new_mean = mean + (x_new - xs[next_index])/window_size;
                # varSum = var_sum + (x_new - mean) * (x_new - new_mean) - (xs[next_index] - mean) * (xs[next_index] - new_mean);

                last_index = (self.window_index+1) % self.window_size
                yold = self.window_buffer[last_index]
                self.mean  = self.mean + y / self.window_size - yold / self.window_size
                # self.sos = self.sos + (y + yold - mean_past - self.mean) * (y - yold)
                self.sos = self.sos + np.dot((y - mean_past), np.transpose((y - self.mean))) - np.dot((yold - mean_past), np.transpose((yold - self.mean)))
                # overwrite the oldest element
                self.window_buffer[last_index] = y

            self.window_index += 1
            self.window_index %= self.window_size

            #print ("m, sos: ", self.mean.shape, self.sos.shape)
            if self.steps > 1:
                if self.steps <= self.window_size:
                    var = self.sos / self.steps #(self.steps-1)
                    # print ("No window var: ", var, self.mean)
                else:
                    var = self.sos / self.steps #(self.window_size-1)
                    # print ("Window var: ", var, self.mean)
                # print ("var: ", var)
                # input("")
                self.Rt = var

        Et = y - self.xt

        # NOTE: this is being computed properly but was being overwhelmed by magnitude of Pt
        if self.use_diagonal_approx:
            # print (self.e, self.Pt, self.Rt)
            # import time
            # start = time.time()
            # import pdb
            # print (self.Pt, self.Rt)
            Kt = self.Pt * 1.0/(self.Pt + self.Rt + 1e-8)
            self.Pt = (self.ones - Kt) * self.Pt
            self.xt = self.xt + Kt * Et

            # print ("pre error: ", self.e)
            self.e = (self.ones - Kt) * self.e
            # print ("post error: ", self.e, Kt, self.Pt, self.Rt)
            self.Kt = Kt
            # print (Kt, self.xt)
            # print (Et)
            # print (self.Pt)
            # print (self.Rt)
            # input("")
            #
            # if self.n == 10:
            #     pdb.set_trace()

            # print (self.e, Kt, self.Pt, self.Rt)
            # input("")
            # end = time.time()
            # print ("Time for kalman update: ", (end-start))
        else:
            # import pdb
            # This may be simply too slow for practical usage...
            # import time
            # start = time.time()
            # print (self.Pt)
            # print (self.Rt)
            # Kt = self.Pt @ linalg.inv(self.Pt + self.Rt + 1e-8)
            Kt = self.Pt @ np.linalg.pinv(self.Pt + self.Rt + 1e-8)
            self.Pt = (self.I - Kt) @ self.Pt
            self.xt = self.xt + Kt @ Et
            self.e = (self.I - Kt) @ self.e
            # if self.n == 2:
            #     pdb.set_trace()
            # print (Kt)
            # print (Et)
            # print (self.Pt)
            # print (self.Rt)
            # input("")
            # print (np.linalg.norm(self.e))
            # end = time.time()
            # print ("Time for kalman update: ", (end-start))

        # print (np.linalg.norm(self.e), Kt, self.Pt, self.Rt)
        # input("")
