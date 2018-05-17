import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import normal, multivariate_normal

class GaussianModel(nn.Module):

    def __init__(self, ndim, mu0=1.0, log_std0=0.0):
        super(GaussianModel, self).__init__()
        self.ndim = ndim
        self.mu0 = mu0
        self.log_std0 = log_std0
        self.mu = nn.Parameter(torch.FloatTensor(ndim).fill_(mu0))
        # self.log_std = nn.Parameter(torch.FloatTensor(ndim).fill_(log_std0))
        self.std = np.ones((ndim,)) * 0.5
        self.nparam = ndim # 2 * ndim

    def unflatten_grad(self, grad):
        # take gradient and reshape back into param matrices
        i = 0
        for p in self.parameters():
            sh = p.shape
            s = i
            e = i + p.numel()
            # get next bunch and make proper shape
            p.grad = grad[s:e].view(p.shape).detach()
            # print (p.grad)
            i = e

    def flattened_grad(self):
        """
        Take the gradients in the .grad attribute of each param and flatten it into a vector
        """
        g = torch.FloatTensor(self.nparam)
        ind = 0
        for p in self.parameters():
            gnumel = p.grad.data.numel()
            g[ind:ind+gnumel] = p.grad.data.view(-1).detach()
            ind += gnumel
        # reshape to nx1 vector
        g = g.view(-1, 1)
        return g

    def forward(self):
        """
        This gives x, log p(x)
        """
        #dist1 = multivariate_normal.MultivariateNormal(self.mu, torch.diag(self.log_std.exp()))
        dist = normal.Normal(self.mu, Variable(torch.from_numpy(self.std).float())) #self.log_std.exp())
        sample = dist.sample()
        #sample2 = dist2.sample()
        #temp_sample = torch.FloatTensor(sample.shape)
        #temp_sample.fill_(sample[0])
        #logp = dist.log_prob(temp_sample)
        #print ("sample:",sample1, sample2)
        logp = dist.log_prob(sample)
        #logp2 = dist2.log_prob(sample2)
        #print ("logp: ", logp1, logp2, torch.mean(logp2), torch.sum(logp2))
        #input("")
        return sample, torch.mean(logp)

class SingleParameterModel(object):
    """
    This class does not use backprop to compute grads, hence no PyTorch
    """

    def __init__(self, ndim, x0=1.0):
        self.ndim = ndim
        self.x0 = x0
        self.x = np.empty((ndim, 1))
        self.x.fill(x0)
        print (self.x.shape)
        self.nparam = ndim
