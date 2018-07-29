import os
import numpy as np
import torch
from torch.autograd import Variable

from rl_adaptive_sampling.rl.utils import simple_sampler, eval

def compute_rollout_grad(args, tgrads, baseline, retain=False):
    empirical_state_value = []

    loss = 0.0
    total_len = 0
    for rewards, logps, states in tgrads:
        rewards = np.array(rewards)
        # rewards /= np.std(rewards) + 1e-8
        # rewards -= np.mean(rewards)
        # discounted = []
        advantages = []
        r = 0.0
        v_next = 0.0 # last value is zero since terminal and operating over trajectories
        gae = 0.0
        for i in reversed(range(len(rewards))):
            r = rewards[i] + args.gamma * r
            empirical_state_value.append((states[i], r))
            # Compute A(s) = Q(s, a) - V(s)
            # Q(s, a) is empirical return. V(s) is approximation
            # Generalized Advantage Estimataion
            v_predict = baseline.predict(states[i])
            delta_t = rewards[i] + args.gamma * v_next - v_predict
            gae = gae * args.gamma * args.tau + delta_t
            v_next = v_predict
            # advantage = r - v_predict
            advantages.append(gae)

        advantages = list(reversed(advantages))
        advantages = np.array(advantages)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # discounted = list(reversed(discounted))
        # discounted = np.array(discounted)
        # discounted = (discounted - np.mean(discounted)) / (np.std(discounted) + 1e-8)

        # print (discounted)
        # input("")
        for i in range(len(advantages)):
            loss += logps[i] * Variable(torch.FloatTensor([advantages[i]]))
        total_len += len(rewards)
        # input("")
    loss = -loss / total_len
    # loss = loss / total_len
    loss.backward(retain_graph=retain)

    return empirical_state_value


def execute_kf_grad_step(model, kf):
    # get grad from KF,
    ghat = kf.xt
    # unflatten
    # print ("KF xt: ", kf.xt)
    gtensor = torch.from_numpy(ghat).float().view(model.lin.weight.shape)
    # set to model params grad
    # print (model.lin.weight.grad)
    model.lin.weight.grad = gtensor.clone()
    # print (model.lin.weight.grad)
    # input("")
