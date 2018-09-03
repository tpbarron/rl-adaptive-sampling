import numpy as np
import scipy.stats as ss
import torch
import torch.nn as nn
from torch.distributions import normal
from torch.autograd import Variable
# import torch.nn.init as init

class LinearPolicy(nn.Module):

    def __init__(self, input_dim, action_dim):
        super(LinearPolicy, self).__init__()
        print ("Linear policy: ", input_dim, action_dim)
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.lin = nn.Linear(input_dim, action_dim, bias=True)
        # for param in list(self.lin.parameters()):  # only last layer
            # param.data = 1e-2 * param.data
        nn.init.kaiming_normal_(self.lin.weight.data)

        self.std = np.ones((action_dim,)) * 0.5

    def num_params(self):
        return self.input_dim * self.action_dim

    def forward(self, x):
        x = x.view(1, -1).float()
        mu = self.lin(x)
        return mu

    def act(self, state, deterministic=False):
        x = self(state)
        if deterministic:
            return x, 0.0
        n = normal.Normal(x, Variable(torch.from_numpy(self.std).float()))
        a = n.sample()
        logp = n.log_prob(a).mean()
        return a, logp
