"""
Linear Natural Policy Gradient
"""

import sys
import os
import csv
import joblib

import numpy as np
import gym
import pybullet_envs
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import categorical, normal, multivariate_normal

from rl_adaptive_sampling.opt.npg_opt import NaturalSGD
from rl_adaptive_sampling.opt.kalman_opt import KalmanFilter
from rl_adaptive_sampling.rl.running_state import ZFilter

class Policy(nn.Module):

    def __init__(self, n_inputs, n_outputs, is_continuous):
        super(Policy, self).__init__()
        # policy
        self.fc1 = nn.Linear(n_inputs, n_outputs)
        # self.fc2 = nn.Linear(128, n_outputs)
        if is_continuous:
            self.log_std = nn.Parameter(torch.zeros(n_outputs))
        nn.init.xavier_normal_(self.fc1.weight.data)
        # nn.init.xavier_normal_(self.fc2.weight.data)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(F.relu(self.fc1(x)))
        return x


class Value(nn.Module):

    def __init__(self, n_inputs):
        super(Value, self).__init__()
        # value approx
        self.fc1v = nn.Linear(n_inputs, 1)
        # self.fc2v = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fc1v.weight.data)
        # nn.init.xavier_normal_(self.fc2v.weight.data)

    def forward(self, x):
        x = self.fc1v(x)
        # x = self.fc2v(F.relu(self.fc1v(x)))
        return x


class FFPolicy(nn.Module):

    def __init__(self, env):
        super(FFPolicy, self).__init__()
        self.env = env
        self.is_continuous = isinstance(env.action_space, spaces.Box)
        n_inputs = env.observation_space.shape[0]
        if self.is_continuous:
            n_outputs = env.action_space.shape[0]
        else:
            n_outputs = env.action_space.n

        self.pi = Policy(n_inputs, n_outputs, self.is_continuous)
        self.v = Value(n_inputs)

    def forward(self, state):
        state = state.view(-1, self.env.observation_space.shape[0])
        # state = self.kernel.forward(state)
        x = self.pi(state)
        v = self.v(state)
        return x, v

    def act(self, state, deterministic=False):
        x, v = self(state)
        if self.is_continuous:
            if deterministic:
                action = x
                action_log_prob = None
                entropy = None
            else:
                c = normal.Normal(x, self.pi.log_std.exp())
                action = c.sample()
                action_log_prob = c.log_prob(action).mean()
                entropy = c.entropy()
        else: # discrete
            if deterministic:
                action = torch.max(F.log_softmax(x, dim=1), dim=1)[1]
                action_log_prob = None
                entropy = None
            else:
                c = categorical.Categorical(logits=F.log_softmax(x, dim=1))
                action = c.sample()
                action_log_prob = c.log_prob(action)
                entropy = c.entropy()
        return action, action_log_prob, v, entropy


def get_num_params(model):
    numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return numel


def get_flattened_grad(opt, model):
    numel = get_num_params(model)
    g = torch.FloatTensor(numel)
    ind = 0
    # print (opt.param_groups[0]['params'])
    for p in opt.param_groups[0]['params']: # only use policy params
        gnumel = p.grad.data.numel()
        g[ind:ind+gnumel] = p.grad.data.view(-1)
        ind += gnumel
    # reshape to nx1 vector
    g = g.view(-1, 1).numpy()
    return g

def compute_grad(action_log_probs, rewards, normalize=False):
    loss = 0.0
    for i in range(len(action_log_probs)):
        loss = loss - action_log_probs[i] * Variable(torch.FloatTensor([rewards[i]]))
    if normalize:
        loss = loss / len(action_log_probs)
    loss.backward(retain_graph=True)

def optimize():
    env = gym.make('Acrobot-v1')
    model = FFPolicy(env)
    opt = optim.SGD(model.pi.parameters(), lr=0.1)

    obs = env.reset()
    done = False
    action_log_probs = []
    rewards = []
    while not done:
        action, action_log_prob, v, entropy = model.act(Variable(torch.from_numpy(obs).float()))
        obs, rew, done, info = env.step(np.squeeze(action.numpy()))
        action_log_probs.append(action_log_prob)
        rewards.append(np.random.random())#rew)

    # print (len(action_log_probs))
    opt.zero_grad()
    compute_grad(action_log_probs, rewards, normalize=True)
    full_grad = get_flattened_grad(opt, model.pi)

    partial_grads = []
    for i in range(0, len(action_log_probs), 2):
        print (i)
        opt.zero_grad()
        # print (len(action_log_probs[i:i+2]))
        compute_grad(action_log_probs[i:i+2], rewards[i:i+2])
        partial_grad = get_flattened_grad(opt, model.pi)
        partial_grads.append(partial_grad)

    print (full_grad.shape)

    partial_grads = np.array(partial_grads)
    print (partial_grads.shape)
    summed_partial_grads = np.sum(partial_grads, axis=0)
    print (summed_partial_grads.shape)
    normed_summed_partial_grads = summed_partial_grads / len(action_log_probs)

    print ("Close? ", np.allclose(full_grad, summed_partial_grads))
    print ("Close? ", np.allclose(full_grad, normed_summed_partial_grads))



if __name__ == '__main__':
    optimize()
