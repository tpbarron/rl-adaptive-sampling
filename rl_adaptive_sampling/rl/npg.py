"""
Linear Natural Policy Gradient
"""

import argparse
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

parser = argparse.ArgumentParser(description='RL Pol Grad')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='learning rate (default: 1e-2)')
parser.add_argument('--batch-size', type=int, default=1000,
                    help='training batch size (default: 1000)')
parser.add_argument('--max-samples', type=int, default=1e6,
                    help='maximum num steps to take (default: 1e6)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.97,
                    help='GAE parameter (default: 0.97)')
parser.add_argument('--kf-error-thresh', type=float, default=0.025,
                    help='threshold for update expected error (default: 0.025)')
parser.add_argument('--env-name', type=str, default='CartPole-v0',
                    help='env to train on (default: CartPole-v0)')
parser.add_argument('--log-dir', type=str, default='/tmp/rl_kalman/',
                    help='dir to save logs (default: /tmp/rl_kalman/)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--no-kalman', action='store_true', default=False,
                    help='do not use kf estimate (default: false)')
args = parser.parse_args()

os.makedirs(args.log_dir, exist_ok=True)
joblib.dump(args, os.path.join(args.log_dir, 'args_snapshot.pkl'))

log_file = open(os.path.join(args.log_dir, 'log.csv'), 'w')
log_writer = csv.writer(log_file)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

env = gym.make(args.env_name)
train_rewards = []
eval_rewards = []
max_reward, avg_reward, total_samples = -np.inf, 0.0, 0.0

zfilter = ZFilter(env.observation_space.shape)

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


def eval(model, avgn=5, render=False):
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
    eval_rewards.append(eval_reward)
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

def compute_grad_log_pi(opt, model, action_log_prob):
    opt.zero_grad()
    action_log_prob.backward(retain_graph=True)
    return get_flattened_grad(opt, model)

def train(model, opt, opt_v, kf, ep=0):
    global max_reward, avg_reward, total_samples
    nparams = get_num_params(model.pi)
    model.train()

    if not args.no_kalman:
        # make as large as max batch
        gae_est = np.zeros((args.batch_size, 1))
        gae_coef = np.zeros((args.batch_size, 1))
        grad_log_pi = np.zeros((args.batch_size, nparams))
        grad_log_pi_adv = np.zeros((args.batch_size, args.batch_size, nparams))
        kalman_errors = np.zeros((args.batch_size, nparams))
        kalman_variances = np.zeros((args.batch_size, nparams))
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
    if not args.no_kalman:
        kf.reset()

    step = 0
    while (not args.no_kalman and np.mean(kf.e) > args.kf_error_thresh or step < 20) and step < args.batch_size:
    # while np.linalg.norm(kf.e)**2.0/nparams > args.kf_error_thresh and step < args.batch_size:
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

        if not args.no_kalman:
            # print (action_log_prob)
            grad_log_pi[step,:] = compute_grad_log_pi(opt, model.pi, action_log_prob)[:,0]
            # print (grad_log_pi[step,:])
            # input("")
            # opt.update_fisher(action_log_prob)

            if step > 0:
                # since don't want to do the extra work to compute V(s_t+1) at every step
                # this will lag one step behind
                # multiply coefs by decay
                # if episode ended at step-1, then zero out gae_coef
                if masks[step-1] == 0: # this means done
                    gae_coef[:step-1] = 0
                    last_reset = step
                    # wait until the next step to start updates again
                else:
                    gae_coef = gae_coef * (args.gamma * args.tau)
                    # set next state to begin accumulating gae
                    gae_coef[step-1] = 1
                    # update all states gae_ests (many are zeroed out) but nicely vectorized
                    delta_t = -ep_values[step-1].data.numpy() + ep_rewards[step-1] + ep_values[step].data.numpy()
                    gae_est = gae_est + gae_coef * delta_t
                    # compute actual grads
                    grad_log_pi_adv[step-1,:] = -grad_log_pi * gae_est

                    # print ("Updating num steps: ", step - last_reset)
                    # only iterate through step because advantages only computed through step-1
                    for k in reversed(range(last_reset, step)):
                        # print (grad_log_pi_adv[step-1,k,:][:,np.newaxis].shape)
                        # print (kf.Pt)
                        kf.update(grad_log_pi_adv[step-1,k,:][:,np.newaxis])
                    # kf.update(grad_log_pi_adv[step-1,step-1,:])
                        # if np.any(np.isnan(kf.Pt)):
                        #     print (grad_log_pi_adv[step-1,k,:][:,np.newaxis])
                        #     print ("kt: ", kf.Kt)
                        #     print ("rt: ", kf.Rt)
                        #     print ("sos: ", kf.sos)
                        #     print ("mean: ", kf.mean)
                        #     input("")

                kalman_errors[step-1,:] = kf.e[:,0]
                kalman_variances[step-1,:] = kf.Rt[:,0]

        traj_rewards += reward
        step += 1

        if done:
            # print ("Episode reward: ", traj_rewards)
            train_rewards.append(traj_rewards)
            if traj_rewards > max_reward:
                max_reward = traj_rewards
            avg_reward = sum(train_rewards[-10:]) / len(train_rewards[-10:])
            # print (max_reward, avg_reward)
            sys.stdout.write("Training: max reward: %f, window (10) average reward: %f \r" % (max_reward, avg_reward))
            sys.stdout.flush()

            if not args.no_kalman:
                # compute last advantage for accumulation
                state = Variable(torch.from_numpy(obs)).float()
                with torch.no_grad():
                    x, value = model(state)

                gae_coef = gae_coef * (args.gamma * args.tau)
                if step < args.batch_size:
                    # set next state to begin accumulating gae
                    gae_coef[step] = 1
                # update all states gae_ests (many are zeroed out) but nicely vectorized
                # print (len(ep_values), step)
                # input("")
                delta_t = -ep_values[step-1].data.numpy() + ep_rewards[step-1] + value.data.numpy()
                gae_est = gae_est + gae_coef * delta_t
                # compute actual grads
                grad_log_pi_adv[step-1,:] = -grad_log_pi * gae_est
                # only iterate through step because advantages only computed through step-1
                for k in reversed(range(last_reset, step)):
                    kf.update(grad_log_pi_adv[step-1,k,:][:,np.newaxis])

            traj_rewards = 0
            obs = env.reset()

            # if not on final step, reset done
            if step != args.batch_size:
                done = False

    total_samples += step
    if not args.no_kalman:
        # print ("")
        sys.stdout.write("\n")
        # print (kf.Pt, kf.Rt)
        print ("Updating after ", step, " steps; total samples: ", total_samples, " with error ", np.mean(kf.e))
        # save data; don't worry about saving grad_grad_log_prob_adv since c
        # computable from other data and saves time at runtime
        np.save(os.path.join(args.log_dir, 'grad_log_pi'+str(ep)+'.npy'), grad_log_pi[0:step,:])
        np.save(os.path.join(args.log_dir, 'adv_est'+str(ep)+'.npy'), gae_est[:step])
        # np.save(os.path.join(args.log_dir, 'grad_log_pi_adv'+str(ep)+'.npy'), grad_log_pi_adv[0:step,0:step,:])
        np.save(os.path.join(args.log_dir, 'kalman_errors'+str(ep)+'.npy'), kalman_errors[0:step,:])
        np.save(os.path.join(args.log_dir, 'kalman_variances'+str(ep)+'.npy'), kalman_variances[0:step,:])

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

    ep_gaes = []
    ep_values.append(Variable(R))
    R = Variable(R)
    gae = torch.zeros(1, 1)
    for i in reversed(range(len(ep_rewards))):
        R = args.gamma * R * masks[i+1] + ep_rewards[i]
        advantage = R - ep_values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)
        # GAE
        delta_t = ep_rewards[i] + args.gamma * ep_values[i + 1].data * masks[i + 1] - ep_values[i].data
        gae = gae * args.gamma * args.tau * masks[i + 1] + delta_t
        ep_gaes.append(float(gae.data))
        # print (ep_action_log_probs[i].data.numpy(), gae.numpy())
        if args.no_kalman:
            policy_loss = policy_loss - ep_action_log_probs[i] * Variable(gae) #- 0.01 * ep_entropies[i]

    policy_loss = policy_loss / len(ep_rewards)
    # add lr reg loss to see if helps stability
    # for weight in model.pi.parameters():
    #     policy_loss = policy_loss + 0.0001 * weight.norm(2)
    value_loss = value_loss / len(ep_rewards)

    # update policy
    if args.no_kalman:
        opt.zero_grad()
        # print ("Policy loss: ", policy_loss.data.numpy(), len(ep_rewards))
        policy_loss.backward()
    else:
        set_grad(opt, model.pi, kf.xt)
    opt.step()

    v_inputs = torch.from_numpy(np.array(ep_states)).float()
    v_targets = torch.from_numpy(np.array(ep_gaes)).float().view(-1, 1)
    def value_opt_closure():
        opt_v.zero_grad()
        out = model.v(v_inputs)
        loss = F.mse_loss(out, v_targets)
        loss.backward()
        return loss

    # update value fn
    opt_v.zero_grad()
    value_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.v.parameters(), 20.0)
    loss = opt_v.step(value_opt_closure)
    # print ("Vinput: ", v_inputs.shape, v_targets.shape, loss.data)
    # input("")

def main():
    model = FFPolicy(env)
    # opt_v = optim.Adam(model.v.parameters(), lr=args.lr)
    opt_v = optim.LBFGS(model.v.parameters(), lr=args.lr)
    opt = NaturalSGD(model.pi.parameters(), lr=args.lr)
    kf = KalmanFilter(state_dim=get_num_params(model.pi), use_last_error=False, use_diagonal_approx=True, error_init=1.0, sos_init=1000.0)

    best_eval = 0
    last_save_step = 0
    e = 0
    while total_samples < args.max_samples:
        train(model, opt, opt_v, kf, ep=e)
        avg_eval = eval(model) #, render=True)
        log_writer.writerow([total_samples, max_reward, avg_reward, avg_eval])
        log_file.flush()
        e += 1
        print ("total samples: ", total_samples)

        if avg_eval > best_eval or last_save_step - total_samples > 10000:
            best_eval = avg_eval
            last_save_step = total_samples
            # save model if evaluation was better
            torch.save(model, os.path.join(args.log_dir, "model_ep"+str(e)+"_samples"+str(total_samples)+"_eval"+str(avg_eval)+".pth"))
    log_file.close()


if __name__ == '__main__':
    main()
