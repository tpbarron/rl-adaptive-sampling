import numpy as np
import torch
from torch.autograd import Variable

def do_rollout(env, pi, optimal=False, eval=False, render=False):
    done = False
    obs = env.reset()
    rewards = []
    logps = []
    states = [obs]

    while not done:
        if render:
            env.render()
        act, logp = pi.act(Variable(torch.from_numpy(obs)), deterministic=eval)
        # print ("Action: ", act.data.numpy(), obs)
        act = np.squeeze(act.data.numpy())
        act = np.clip(act, env.action_space.low, env.action_space.high)
        obs, rew, done, info = env.step(act) #, use_k=optimal)
        rewards.append(rew)
        logps.append(logp)
        states.append(obs)
        if render:
            import time
            time.sleep(0.01)
    return rewards, logps, states
