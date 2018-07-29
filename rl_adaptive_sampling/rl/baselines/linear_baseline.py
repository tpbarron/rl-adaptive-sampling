# From: https://github.com/aravindr93/mjrl/blob/master/mjrl/baselines/linear_baseline.py
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.kernel_ridge import KernelRidge

class LinearBaselineKernelRegression:
    def __init__(self, env, **kwargs):
        n = env.observation_space.shape[0]  # number of states
        self.fitted = False
        self.krrm = KernelRidge(alpha=1.0, kernel='rbf')

    def fit(self, empirical_state_value):
        states, values = zip(*empirical_state_value)
        states = np.stack(states)
        self.krrm.fit(states, values)

    def predict(self, state):
        if not self.fitted:
            return 0.0
        return self.krrm.predict(state)

class LinearBaselineParameterized(nn.Module):
    def __init__(self, env, **kwargs):
        super(LinearBaselineParameterized, self).__init__()
        n = env.observation_space.shape[0]  # number of states
        a = env.action_space.shape[0]
        self.fc1 = nn.Linear(n, a)
        self.opt = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, state):
        return self.fc1(state)

    def fit(self, empirical_state_value):
        states, values = zip(*empirical_state_value)
        states = np.stack(states)
        values = np.array(values)

        self.opt.zero_grad()
        inp_state = Variable(torch.from_numpy(states).float())
        output = self.forward(inp_state)
        target = Variable(torch.from_numpy(values).float()).view(-1, 1)
        # print (output.shape)
        # input("")
        # print (output, target)
        loss = F.mse_loss(output, target)
        print ("Loss: ", loss.data.numpy() / len(states))
        loss.backward()
        self.opt.step()

    def predict(self, state):
        pred = self.forward(Variable(torch.from_numpy(state).float())).detach().numpy()
        # print (pred)
        return pred
