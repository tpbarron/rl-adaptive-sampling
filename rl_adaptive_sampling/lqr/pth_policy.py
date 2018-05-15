import numpy as np
import scipy.stats as ss
import torch
import torch.nn as nn
from torch.distributions import normal
from torch.autograd import Variable

class LinearPolicy(nn.Module):

    def __init__(self, input_dim, action_dim):
        super(LinearPolicy, self).__init__()
        print ("Linear policy: ", input_dim, action_dim)
        self.lin = nn.Linear(input_dim, action_dim)

        self.std = np.ones((action_dim,)) * 0.5

    def forward(self, x):
        x = x.view(1, -1).float()
        x = self.lin(x)
        return x

    def act(self, state):
        x = self(state)
        n = normal.Normal(x, Variable(torch.from_numpy(self.std).float()))
        a = n.sample()
        logp = n.log_prob(a).mean()
        return a, logp
