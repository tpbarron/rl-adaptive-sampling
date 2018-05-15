import numpy as np
import scipy.stats as ss

class GaussianModel(object):

    def __init__(self, ndim, mu0=1.0, log_std0=0.0):
        super(GaussianModel, self).__init__()
        self.ndim = ndim
        self.mu0 = mu0
        self.log_std0 = log_std0
        self.mu = np.empty((self.ndim,))
        self.mu.fill(mu0)
        self.std = np.ones((self.ndim,)) * 0.1
        self.nparam = ndim # 2 * ndim # mean only

    def sample(self):
        """
        This gives x, log p(x)
        """
        sample = np.random.normal(self.mu, self.std)
        logp = ss.norm.logpdf(sample, loc=self.mu, scale=self.std)
        return sample, np.mean(logp)


class SingleParameterModel(object):
    """
    """

    def __init__(self, ndim, x0=1.0):
        self.ndim = ndim
        self.x0 = x0
        self.x = np.empty((ndim, 1))
        self.x.fill(x0)
        self.nparam = ndim
