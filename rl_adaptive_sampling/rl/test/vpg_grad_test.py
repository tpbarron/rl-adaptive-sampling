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

grads = []
true_grad = None

def train(args, env, model, opt, opt_v, stats, ep=0):
    nparams = get_num_params(model.pi)
    global grads, true_grad
    model.train()

    traj_reward = 0

    # hold action vars and probs
    ep_states = []
    ep_actions = []
    ep_action_log_probs = []
    ep_entropies = []
    ep_values = []
    ep_rewards = []

    traj_action_log_probs = []
    traj_values = []
    traj_rewards = []
    traj_entropies = []

    done = False
    obs = env.reset()

    traj_step = 0
    step = 0
    while True:
        ep_states.append(obs)
        state = Variable(torch.from_numpy(obs)).float()
        action, action_log_prob, value, entropy = model.act(state)

        action_np = action.data.numpy().squeeze()
        if model.is_continuous:
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
        obs, reward, done, _ = env.step(action_np)

        traj_values.append(value)
        traj_rewards.append(reward)
        traj_action_log_probs.append(action_log_prob)
        traj_entropies.append(entropy)

        traj_reward += reward
        step += 1
        traj_step += 1

        if done:
            # print ("Episode reward: ", traj_rewards)
            stats['train_rewards'].append(traj_reward)
            if traj_reward > stats['max_reward']:
                stats['max_reward'] = traj_reward
            stats['avg_reward'] = sum(stats['train_rewards'][-10:]) / len(stats['train_rewards'][-10:])

            # compute GAE and update KF
            R = torch.zeros(1, 1)
            traj_values.append(Variable(R))
            R = Variable(R)
            gae = torch.zeros(1, 1)
            policy_loss = 0.0
            for i in reversed(range(len(traj_rewards))):
                R = args.gamma * R + traj_rewards[i]
                advantage = R - traj_values[i]
                delta_t = traj_rewards[i] + args.gamma * traj_values[i + 1].data - traj_values[i].data
                gae = gae * args.gamma * args.tau + delta_t
                policy_loss = policy_loss - traj_action_log_probs[i] * Variable(gae) #- 0.01 * ep_entropies[i]

            # print (np.mean(np.asarray(mean_batch_sizes)))
            # mean_bs = np.mean(np.asarray(mean_batch_sizes))
            # policy_loss = policy_loss / mean_bs
            opt.zero_grad()
            policy_loss.backward(retain_graph=True)
            del traj_values[-1]
            grad = get_flattened_grad(opt, model.pi)
            grads.append(grad)
            # print ("GRAD: ", grad.shape, len(ep_rewards), traj_step, np.mean(kf.e))
            # input("")
            print ("traj length: ", traj_step)

            traj_reward = 0
            traj_step = 0

            ep_values.append(np.asarray(traj_values))
            ep_rewards.append(np.asarray(traj_rewards))
            ep_action_log_probs.append(np.asarray(traj_action_log_probs))
            ep_entropies.append(np.asarray(traj_entropies))

            del (traj_action_log_probs[:])
            del (traj_values[:])
            del (traj_rewards[:])
            del (traj_entropies[:])

            for i in range(len(ep_rewards)):
                print (ep_rewards[i])
            input("")

            if step >= args.batch_size:
                break
            elif step < args.batch_size:
                # if not on final step, reset done
                done = False
                obs = env.reset()


    policy_loss = 0
    value_loss = 0

    print ("ntrajs: ", len(ep_rewards))

    for j in range(len(ep_rewards)): # loop over trajs
        traj_values = list(ep_values[j])
        print (len(traj_values))
        traj_rewards = ep_rewards[j]
        print (traj_rewards)
        input("")
        traj_action_log_probs = ep_action_log_probs[j]
        R = torch.zeros(1, 1)
        traj_values.append(Variable(R))
        R = Variable(R)
        gae = torch.zeros(1, 1)
        traj_pol_loss = 0
        for i in reversed(range(len(traj_rewards))):
            R = args.gamma * R + traj_rewards[i]
            advantage = R - traj_values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            delta_t = traj_rewards[i] + args.gamma * traj_values[i + 1].data - traj_values[i].data
            gae = gae * args.gamma * args.tau + delta_t
            policy_loss = policy_loss - traj_action_log_probs[i] * Variable(gae) #- 0.01 * ep_entropies[i]
            traj_pol_loss = traj_pol_loss - traj_action_log_probs[i] * R
            # print (policy_loss.data.numpy(), traj_action_log_probs[o])
        print ("Fullgrad traj:", traj_pol_loss.data.numpy())

    policy_loss = policy_loss / step
    print ("len rew: ", step)
    value_loss = value_loss / step

    # update policy
    opt.zero_grad()
    policy_loss.backward()
    true_grad = get_flattened_grad(opt, model.pi)
    opt.step()

    # update value fn
    opt_v.zero_grad()
    value_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.v.parameters(), 20.0)
    loss = opt_v.step()

    stats['total_samples'] += step
    return step


def optimize(args):
    global grads, true_grad

    env = gym.make('CartPole-v0')
    model = FFPolicy(env)
    opt = optim.SGD(model.pi.parameters(), lr=0.1)
    opt_v = optim.SGD(model.v.parameters(), lr=0.1)

    stats = {}
    stats['train_rewards'] = []
    stats['eval_rewards'] = []
    stats['max_reward'] = -np.inf
    stats['avg_reward'] = 0.0
    stats['total_samples'] = 0.0

    train(args, env, model, opt, opt_v, stats, ep=0)

    print (true_grad.shape)
    partial_grads = np.array(grads)
    print (partial_grads.shape)
    summed_partial_grads = np.sum(partial_grads, axis=0)
    print (summed_partial_grads.shape)
    normed_summed_partial_grads = summed_partial_grads / stats['total_samples']
    print (stats)

    print ("Close? ", np.allclose(true_grad, summed_partial_grads))
    print ("Close? ", np.allclose(true_grad, normed_summed_partial_grads))

    print (true_grad)
    print (normed_summed_partial_grads)

    #
    # obs = env.reset()
    # done = False
    # action_log_probs = []
    # rewards = []
    # while not done:
    #     action, action_log_prob, v, entropy = model.act(Variable(torch.from_numpy(obs).float()))
    #     obs, rew, done, info = env.step(np.squeeze(action.numpy()))
    #     action_log_probs.append(action_log_prob)
    #     rewards.append(np.random.random())#rew)
    #
    # # print (len(action_log_probs))
    # opt.zero_grad()
    # compute_grad(action_log_probs, rewards, normalize=True)
    # full_grad = get_flattened_grad(opt, model.pi)
    #
    # partial_grads = []
    # for i in range(0, len(action_log_probs), 2):
    #     print (i)
    #     opt.zero_grad()
    #     # print (len(action_log_probs[i:i+2]))
    #     compute_grad(action_log_probs[i:i+2], rewards[i:i+2])
    #     partial_grad = get_flattened_grad(opt, model.pi)
    #     partial_grads.append(partial_grad)
    #
    # print (full_grad.shape)
    #
    # partial_grads = np.array(partial_grads)
    # print (partial_grads.shape)
    # summed_partial_grads = np.sum(partial_grads, axis=0)
    # print (summed_partial_grads.shape)
    # normed_summed_partial_grads = summed_partial_grads / len(action_log_probs)
    #
    # print ("Close? ", np.allclose(full_grad, summed_partial_grads))
    # print ("Close? ", np.allclose(full_grad, normed_summed_partial_grads))



if __name__ == '__main__':
    import arguments
    args = arguments.get_args()
    optimize(args)
