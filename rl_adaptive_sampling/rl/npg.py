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
        if is_continuous:
            self.log_std = nn.Parameter(torch.zeros(n_outputs))
        nn.init.xavier_normal_(self.fc1.weight.data)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Value(nn.Module):

    def __init__(self, n_inputs):
        super(Value, self).__init__()
        # value approx
        self.fc1v = nn.Linear(n_inputs, 1)
        nn.init.xavier_normal_(self.fc1v.weight.data)

    def forward(self, x):
        x = self.fc1v(x)
        return x


class FFPolicy(nn.Module):

    def __init__(self, env):
        super(FFPolicy, self).__init__()

        self.is_continuous = isinstance(env.action_space, spaces.Box)
        n_inputs = env.observation_space.shape[0]
        if self.is_continuous:
            n_outputs = env.action_space.shape[0]
        else:
            n_outputs = env.action_space.n

        self.pi = Policy(n_inputs, n_outputs, self.is_continuous)
        self.v = Value(n_inputs)

    def forward(self, state):
        state = state.view(1, -1)
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
                action_log_prob = c.log_prob(action).sum()
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


def eval(args, env, model, stats, avgn=5, render=False):
    model.eval()
    if render:
        env.reset()
        env.render()

    eval_reward = 0.0
    for i in range(avgn):
        done = False
        obs = env.reset()
        while not done:
            if render:
                env.render()
            # features = model.get_features(obs)
            state = Variable(torch.from_numpy(obs)).float()
            action, action_log_prob, value, entropy = model.act(state, deterministic=False)

            action_np =action.data.numpy().squeeze()
            # action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
            obs, reward, done, _ = env.step(action_np)
            eval_reward += reward

    eval_reward /= avgn
    stats['eval_rewards'].append(eval_reward)
    sys.stdout.write("\r\nEval reward: %f \r\n" % (eval_reward))
    sys.stdout.flush()

    return eval_reward


def compute_grad_log_pi(opt, model, action_log_prob):
    opt.zero_grad()
    action_log_prob.backward(retain_graph=True)
    return get_flattened_grad(opt, model)


def train(args, env, model, opt, opt_v, stats, ep=0):
    model.train()

    traj_rewards = 0
    last_reset = 0

    # hold action vars and probs
    ep_states = []
    ep_actions = []
    ep_action_log_probs = []
    ep_entropies = []
    ep_values = []
    ep_rewards = []
    masks = [1]

    done = False
    obs = env.reset()

    step = 0
    while step < args.batch_size:
        # obs = zfilter(obs)
        # print ("obs: ", obs)
        ep_states.append(obs)
        state = Variable(torch.from_numpy(obs)).float()
        action, action_log_prob, value, entropy = model.act(state)

        action_np = action.data.numpy().squeeze()
        # action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
        obs, reward, done, _ = env.step(action_np)

        ep_values.append(value)
        ep_rewards.append(reward)
        ep_actions.append(action)
        ep_action_log_probs.append(action_log_prob)
        ep_entropies.append(entropy)
        masks.append(1 if not done else 0)

        traj_rewards += reward
        step += 1

        if done:
            # print ("Episode reward: ", traj_rewards)
            stats['train_rewards'].append(traj_rewards)
            if traj_rewards > stats['max_reward']:
                stats['max_reward'] = traj_rewards
            stats['avg_reward'] = sum(stats['train_rewards'][-10:]) / len(stats['train_rewards'][-10:])
            # print (max_reward, avg_reward)
            sys.stdout.write("Training: max reward: %f, window (10) average reward: %f \r" % (stats['max_reward'], stats['avg_reward']))
            sys.stdout.flush()

            traj_rewards = 0

            obs = env.reset()
            # if not on final step, reset done
            if step != args.batch_size:
                done = False

    opt.compute_fisher(ep_action_log_probs)

    policy_loss = 0
    value_loss = 0

    R = torch.zeros(1, 1)
    if not done:
        # get last state value est
        state = Variable(torch.from_numpy(obs)).float()
        with torch.no_grad():
            x, value = model(state)
        R = value.data

    ep_returns = []
    ep_values.append(Variable(R))
    R = Variable(R)
    gae = torch.zeros(1, 1)
    for i in reversed(range(len(ep_rewards))):
        R = args.gamma * R * masks[i+1] + ep_rewards[i]
        ep_returns.append(float(R.data))
        advantage = R - ep_values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)
        # GAE
        delta_t = ep_rewards[i] + args.gamma * ep_values[i + 1].data * masks[i + 1] - ep_values[i].data
        gae = gae * args.gamma * args.tau * masks[i + 1] + delta_t
        # print (ep_action_log_probs[i].data.numpy(), gae.numpy())
        policy_loss = policy_loss - ep_action_log_probs[i] * Variable(gae) #- 0.01 * ep_entropies[i]

    policy_loss = policy_loss / len(ep_rewards)
    # add lr reg loss to see if helps stability
    # for weight in model.pi.parameters():
    #     policy_loss = policy_loss + 0.0001 * weight.norm(2)
    value_loss = value_loss / len(ep_rewards)

    opt.zero_grad()
    policy_loss.backward()

    # v_inputs = torch.from_numpy(np.array(ep_states)).float()
    # v_targets = torch.from_numpy(np.array(list(reversed(ep_returns)))).float().view(-1, 1)
    # def value_opt_closure():
    #     opt_v.zero_grad()
    #     out = model.v(v_inputs)
    #     loss = F.mse_loss(out, v_targets)
    #     loss.backward()
    #     return loss

    # update value fn
    opt_v.zero_grad()
    value_loss.backward()
    loss = opt_v.step()#value_opt_closure)

    stats['total_samples'] += step
    return step


def optimize(args):
    print ("Starting variant: ", args)

    args.log_dir = os.path.join(args.log_dir, "env"+args.env_name+"_max_samples"+str(args.max_samples)+"_batch"+str(args.batch_size)+"_lr"+str(args.lr)+"_piopt"+str(args.pi_optim)+"_error"+str(args.kf_error_thresh)+"_diag"+str(int(args.use_diagonal_approx))+"_sos"+str(args.sos_init)+"_resetkfx"+str(int(args.reset_kf_state))+"_resetobs"+str(int(args.reset_obs_noise)))
    args.log_dir = os.path.join(args.log_dir, str(args.seed))
    print ("Starting variant: ", args.log_dir)

    os.makedirs(args.log_dir, exist_ok=True)
    joblib.dump(args, os.path.join(args.log_dir, 'args_snapshot.pkl'))
    log_file = open(os.path.join(args.log_dir, 'log.csv'), 'w')
    log_writer = csv.writer(log_file)

    env = gym.make(args.env_name)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    stats = {}
    stats['train_rewards'] = []
    stats['eval_rewards'] = []
    stats['max_reward'] = -np.inf
    stats['avg_reward'] = 0.0
    stats['total_samples'] = 0.0

    zfilter = ZFilter(env.observation_space.shape)

    model = FFPolicy(env)
    opt_v = optim.Adam(model.v.parameters(), lr=args.lr)
    # opt_v = optim.LBFGS(model.v.parameters(), lr=0.1) #args.lr)
    opt = NaturalSGD(model.pi.parameters(), lr=args.lr)

    best_eval = 0
    last_save_step = 0
    last_iter_samples = 0
    e = 0
    while stats['total_samples'] < args.max_samples:
        train(args, env, model, opt, opt_v, stats, ep=e)
        avg_eval = eval(args, env, model, stats)
        log_writer.writerow([stats['total_samples'], stats['max_reward'], stats['avg_reward'], avg_eval])
        log_file.flush()
        joblib.dump(stats, os.path.join(args.log_dir, 'stats_'+str(e)))
        e += 1
        print ("total samples: ", stats['total_samples'], stats['total_samples']-last_iter_samples)
        last_iter_samples = stats['total_samples']
        if avg_eval > best_eval or last_save_step - stats['total_samples'] > 10000:
            best_eval = avg_eval
            last_save_step = stats['total_samples']
            # save model if evaluation was better
            torch.save(model.state_dict(), os.path.join(args.log_dir, "model_ep"+str(e)+"_samples"+str(stats['total_samples'])+"_eval"+str(avg_eval)+".pth"))
    log_file.close()


if __name__ == '__main__':
    import arguments
    optimize(arguments.get_args())
