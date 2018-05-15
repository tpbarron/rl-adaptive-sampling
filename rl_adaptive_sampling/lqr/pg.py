import numpy as np

import envs.lqg_env as lqg
import envs.lqr_env as lqr

import policy

gamma = 0.99

def do_rollout(env, pi):
    done = False
    obs = env.reset()
    rewards = []
    glogps = []
    states = [obs]

    while not done:
        act, glogp = pi.act(obs)
        obs, rew, done, info = env.step(act)
        rewards.append(rew)
        glogps.append(glogp)
        states.append(obs)
    return rewards, glogps, states


def compute_rollout_grad(rewards, glogps):
    # print ("Grads")
    # compute coefs
    g = np.zeros_like(glogps[0])
    rewards = np.array(rewards)
    rewards /= np.std(rewards) + 1e-8
    rewards -= np.mean(rewards)
    r = 0.0
    for i in reversed(range(len(rewards))):
        r = rewards[i] + gamma * r
        g += glogps[i] * r
    g /= len(rewards)
    return g


def optimize(args):

    env = lqg.LQG_Env()
    pi = policy.LinearPolicy(env.observation_space.shape[0], env.action_space.shape[0])

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
        # print ("Episode return: ", sum(rewards))

        # update policy
        grad = compute_rollout_grad(rewards, glogps)
        tgrads.append(grad)
        last_rewards.append(sum(rewards))

        if (i+1)%100 == 0:
            g = np.stack(tgrads)
            g = np.mean(g, axis=0)
            pi.apply(g)
            tgrads = []

            print ("Avg return:", np.mean(np.array(last_rewards)))
            last_rewards = []
            # for s in states:
            #     print (s)
            #     input("")

        # if (i+1)%100 == 0:

        # pi.apply(grad)
        # input("")
        i += 1
if __name__ == '__main__':
    optimize(0)
