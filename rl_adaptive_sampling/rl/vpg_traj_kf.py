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



def compute_avg_dist(env):
    dists = []
    for i in range(100):
        s = env.reset()
        n, _, d, _ = env.step(env.action_space.sample())
        dist = np.linalg.norm(s-n)
        dists.append(dist)
        if d:
            s = env.reset()
    print ("Avg dist: ", sum(dists) / len(dists))
    input("")

class KernelTransform(object):

    def __init__(self, env, nfeatures=500, bandwidth=1.0):
        self.num_inputs = env.observation_space.shape[0]
        self.P = Variable(torch.from_numpy(np.random.normal(size=(nfeatures, self.num_inputs)))).float()
        self.bandwidth = bandwidth
        self.nfeatures = nfeatures
        self.phi = Variable(torch.from_numpy(np.random.uniform(-np.pi, np.pi, nfeatures))).float()

    def forward(self, state):
        ps = torch.matmul(state, self.P.t()) / self.bandwidth
        f = torch.sin(ps + self.phi)
        return f

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
        #
        # nfeatures = 250
        # n_inputs = nfeatures
        # self.kernel = KernelTransform(env, nfeatures=nfeatures, bandwidth=1.0)

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
            if model.is_continuous:
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
            obs, reward, done, _ = env.step(action_np)
            eval_reward += reward

    eval_reward /= avgn
    stats['eval_rewards'].append(eval_reward)
    sys.stdout.write("\r\nEval reward: %f \r\n" % (eval_reward))
    sys.stdout.flush()

    return eval_reward


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


def unflatten_grad(opt, grad):
    # take gradient and reshape back into param matrices
    ps = []
    i = 0
    # print ("grad: ", grad.shape, grad)
    for p in opt.param_groups[0]['params']:
        # print (p.shape)
        sh = p.shape
        s = i
        e = i + p.numel()
        # get next bunch and make proper shape
        ps.append(torch.from_numpy(grad[s:e]).float().view(p.shape))
        i = e
    return ps


def set_grad(opt, model, x):
    # print (x.shape)
    grads = unflatten_grad(opt, x)
    i = 0
    for p in opt.param_groups[0]['params']: # only use policy params
        p.grad.data = grads[i]
        i += 1

grads = []
true_grad = None

from collections import deque
mean_batch_sizes = deque(maxlen=10)
mean_batch_sizes.append(500)

def train(args, env, model, opt, opt_v, kf, stats, ep=0):
    nparams = get_num_params(model.pi)
    global grads, true_grad
    model.train()

    traj_reward = 0
    last_reset = 0

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
    if not args.no_kalman:
        kf.reset()

    traj_step = 0
    step = 0
    while True:
        # obs = zfilter(obs)
        # print ("obs: ", obs)
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

            if not args.no_kalman:
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

                opt.zero_grad()
                policy_loss.backward(retain_graph=True)
                grad = get_flattened_grad(opt, model.pi)
                grads.append(grad)

                # print (np.mean(np.asarray(mean_batch_sizes)))
                mean_bs = np.mean(np.asarray(mean_batch_sizes))
                policy_loss = policy_loss / mean_bs
                opt.zero_grad()
                policy_loss.backward(retain_graph=True)
                normed_grad = get_flattened_grad(opt, model.pi)
                kf.update(normed_grad)

                del traj_values[-1]

            # print ("traj length: ", traj_step)

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

            if not args.no_kalman and np.mean(kf.e) < args.kf_error_thresh and step > 10:
                # print ("Breaking at error ", np.mean(kf.e))
                break
            elif step >= args.batch_size:
                break
            elif step < args.batch_size:
                # if not on final step, reset done
                done = False
                obs = env.reset()


    policy_loss = 0
    value_loss = 0

    # print ("ntrajs: ", len(ep_rewards))
    for j in range(len(ep_rewards)): # loop over trajs
        traj_values = list(ep_values[j])
        traj_rewards = ep_rewards[j]
        traj_action_log_probs = ep_action_log_probs[j]
        R = torch.zeros(1, 1)
        traj_values.append(Variable(R))
        R = Variable(R)
        gae = torch.zeros(1, 1)
        # traj_pol_loss = 0
        for i in reversed(range(len(traj_rewards))):
            R = args.gamma * R + traj_rewards[i]
            advantage = R - traj_values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            delta_t = traj_rewards[i] + args.gamma * traj_values[i + 1].data - traj_values[i].data
            gae = gae * args.gamma * args.tau + delta_t
            policy_loss = policy_loss - traj_action_log_probs[i] * Variable(gae) #- 0.01 * ep_entropies[i]
            # traj_pol_loss = traj_pol_loss - traj_action_log_probs[i] * R
            # print (policy_loss.data.numpy(), traj_action_log_probs[o])
        # print ("Fullgrad traj:", traj_pol_loss.data.numpy())

    policy_loss = policy_loss / step
    value_loss = value_loss / step

    # update policy
    # opt.zero_grad()
    # policy_loss.backward()
    # true_grad = get_flattened_grad(opt, model.pi)
    # opt.step()

    if args.no_kalman:
        # print ("Policy loss: ", policy_loss.data.numpy(), len(ep_rewards))
        policy_loss.backward()
        # true_grad = get_flattened_grad(opt, model.pi)
        # opt.zero_grad()
        # set_grad(opt, model.pi, true_grad)
    else:
        opt.zero_grad()
        set_grad(opt, model.pi, kf.xt)
    opt.step()

    # update value fn
    opt_v.zero_grad()
    value_loss.backward()
    loss = opt_v.step()

    mean_batch_sizes.append(step)
    stats['total_samples'] += step

    return step


def optimize(args):
    print ("Starting variant: ", args)
    args.log_dir = os.path.join(args.log_dir, "env"+args.env_name+"_max_samples"+str(args.max_samples)+"_batch"+str(args.batch_size)+"_lr"+str(args.lr)+"_piopt"+str(args.pi_optim)+"_error"+str(args.kf_error_thresh)+"_diag"+str(int(args.use_diagonal_approx))+"_sos"+str(args.sos_init)+"_resetkfx"+str(int(args.reset_kf_state)))
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

    # compute_avg_dist(env)
    stats = {}
    stats['train_rewards'] = []
    stats['eval_rewards'] = []
    stats['max_reward'] = -np.inf
    stats['avg_reward'] = 0.0
    stats['total_samples'] = 0.0

    # zfilter = ZFilter(env.observation_space.shape)

    model = FFPolicy(env)
    opt_v = optim.Adam(model.v.parameters(), lr=args.lr)
    # opt_v = optim.LBFGS(model.v.parameters(), lr=args.lr)
    if args.pi_optim == 'adam':
        opt = optim.Adam(model.pi.parameters(), lr=args.lr)
    elif args.pi_optim == 'sgd':
        opt = optim.SGD(model.pi.parameters(), lr=args.lr, momentum=0.1)
    kf = KalmanFilter(state_dim=get_num_params(model.pi), use_last_error=args.use_last_error, use_diagonal_approx=args.use_diagonal_approx, error_init=1.0, sos_init=args.sos_init, reset_state=args.reset_kf_state, reset_observation_noise=False)

    best_eval = 0
    last_save_step = 0
    last_iter_samples = 0
    e = 0
    while stats['total_samples'] < args.max_samples:
        train(args, env, model, opt, opt_v, kf, stats, ep=e)

        global grads, true_grad

        # print (true_grad.shape)
        # partial_grads = np.array(grads)
        # print (partial_grads.shape)
        # summed_partial_grads = np.sum(partial_grads, axis=0)
        # print (summed_partial_grads.shape)
        # normed_summed_partial_grads = summed_partial_grads / stats['total_samples']
        # print (stats)
        #
        # print ("Close? ", np.allclose(true_grad, summed_partial_grads))
        # print ("Close? ", np.allclose(true_grad, normed_summed_partial_grads))
        #
        # print (true_grad)
        # print (normed_summed_partial_grads)
        # input("")

        avg_eval = eval(args, env, model, stats)
        log_writer.writerow([stats['total_samples'], stats['max_reward'], stats['avg_reward'], avg_eval])
        log_file.flush()
        e += 1
        # print ("total samples: ", stats['total_samples'], stats['total_samples']-last_iter_samples)
        last_ep_samples = stats['total_samples']-last_iter_samples
        last_iter_samples = stats['total_samples']
        if avg_eval > best_eval or last_save_step - stats['total_samples'] > 10000:
            best_eval = avg_eval
            last_save_step = stats['total_samples']
            # save model if evaluation was better
            torch.save(model.state_dict(), os.path.join(args.log_dir, "model_ep"+str(e)+"_samples"+str(stats['total_samples'])+"_eval"+str(avg_eval)+".pth"))
        # args.kf_error_thresh = args.kf_error_thresh * 0.995

        # global grads, true_grad
        # if e % 1 == 0:
        #     import matplotlib.pyplot as plt
        #     gs = np.array(grads)
        #     # print ("grads: ", gs.shape)
        #     # input("")
        #     i = 1
        #     for i in range(len(gs[0])):
        #         plt.hist(gs[:,i,0]/last_ep_samples, color='gray')
        #         plt.axvline(np.sum(gs[:,i,0])/last_ep_samples, color='orange')
        #         plt.axvline(kf.xt[i], color='red')
        #         plt.axvline(true_grad[i], color='blue')
        #         # print (gs[:,i,0].shape)
        #         print ("traj, kalman, true: ", np.sum(gs[:,i,0])/last_ep_samples, kf.xt[i], true_grad[i])
        #         plt.show()
        #     # print (len(grads), gs.shape)
        # grads = []
        # input("")
    log_file.close()


if __name__ == '__main__':
    import arguments
    optimize(arguments.get_args())
