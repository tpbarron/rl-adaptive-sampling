import numpy as np

import envs.lqg_env as lqg
import envs.lqr_env as lqr

import pth_policy

import torch
import torch.optim as optim
from torch.autograd import Variable

gamma = 0.99

def do_rollout(env, pi):
    done = False
    obs = env.reset()
    rewards = []
    logps = []
    states = [obs]

    while not done:
        act, logp = pi.act(Variable(torch.from_numpy(obs)))
        obs, rew, done, info = env.step(np.squeeze(act.data.numpy()))
        rewards.append(rew)
        logps.append(logp)
        states.append(obs)
    return rewards, logps, states


def compute_rollout_grad(tgrads):
    loss = 0.0
    total_len = 0
    for rewards, logps in tgrads:
        rewards = np.array(rewards)
        rewards /= np.std(rewards) + 1e-8
        # rewards -= np.mean(rewards)
        r = 0.0
        for i in reversed(range(len(rewards))):
            r = rewards[i] + gamma * r
            loss += logps[i] * Variable(torch.FloatTensor([r]))
        total_len += len(rewards)
    loss = loss / total_len
    loss.backward()


def optimize(args):
    env = lqg.LQG_Env()
    pi = pth_policy.LinearPolicy(env.observation_space.shape[0], env.action_space.shape[0])
    opt = optim.Adam(pi.parameters(), lr=0.01)

    print ("Random episodes")
    for i in range(5):
        rewards, _, _ = do_rollout(env, pi)
        print ("Episode return: ", sum(rewards))
    input("continue?")

    last_rewards = []
    tgrads = []
    i = 0
    while True:
        # sample data
        rewards, glogps, states = do_rollout(env, pi)
        tgrads.append([rewards, glogps])
        # print ("Episode return: ", sum(rewards))

        if (i+1)%10 == 0:
            compute_rollout_grad(tgrads)
            opt.step()
            tgrads = []

        last_rewards.append(sum(rewards))
        if (i+1)%10 == 0:
            print ("Avg return:", np.mean(np.array(last_rewards)))
            last_rewards = []

        i += 1
if __name__ == '__main__':
    optimize(0)
