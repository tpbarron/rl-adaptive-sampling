import numpy as np
from rl_adaptive_sampling.rl.utils import simple_sampler

def eval_pi(env, pi, avg_n=10):
    returns = []
    for i in range(avg_n):
        rewards, _, _ = simple_sampler.do_rollout(env, pi, eval=True, render=False)
        returns.append(sum(rewards))
    avg = np.mean(np.array(returns))
    return avg
