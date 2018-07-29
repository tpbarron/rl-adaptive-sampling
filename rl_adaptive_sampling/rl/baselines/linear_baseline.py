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


class LinearPolynomialKernelBaseline:
    def __init__(self, env, reg_coeff=1e-5):
        n = env.observation_space.shape[0]  # number of states
        self._reg_coeff = reg_coeff
        self._coeffs = None

    def _features(self, obs):
        # compute regression features for the path
        o = np.clip(obs, -10, 10)
        if o.ndim > 2:
            o = o.reshape(o.shape[0], -1)
        l = len(obs)
        al = np.arange(l) / 1000.0
        # print (o.shape, al.shape, l)
        feat = np.concatenate([o, al, al**2, al**3, np.ones((l,))])
        # print ("feat: ", feat.shape)
        return feat

    def fit(self, empirical_state_value):
        states, values = zip(*empirical_state_value)

        # print (len(states), len(values), states[0].shape, len(states[0]))
        # featmat = np.concatenate([self._features(path) for path in paths])
        featmat = np.stack([self._features(state) for state in states])
        # states = np.stack(states)
        returns = np.array(values)
        # print ("featmat: ", featmat.shape, returns.shape)

        reg_coeff = copy.deepcopy(self._reg_coeff)
        for _ in range(10):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10
        # print ("Reg coef: ", self._coeffs)
        # input("")


    def predict(self, state):
        if self._coeffs is None:
            return 0.0
        val = self._features(state).dot(self._coeffs)
        # print ("pred: ", val)
        return val
