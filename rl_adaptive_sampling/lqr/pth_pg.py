import os
import numpy as np

import envs.lqg_env as lqg
import envs.lqr_env as lqr
import envs.cartpole as cp
import envs.linearized_cartpole as lcp

import pth_policy

import torch
import torch.optim as optim
from torch.autograd import Variable

from rl_adaptive_sampling.opt.kalman_opt import KalmanFilter

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

def eval_pi(env, pi, avg_n=10):
    returns = []
    for i in range(avg_n):
        rewards, _, _ = do_rollout(env, pi, eval=True, render=False)
        returns.append(sum(rewards))
    avg = np.mean(np.array(returns))
    return avg

def compute_rollout_grad(args, tgrads, retain=False):
    # TODO: Fix V(s)
    # TODO: Compute GAE baseline
    # TODO: 
    loss = 0.0
    total_len = 0
    for rewards, logps in tgrads:
        rewards = np.array(rewards)
        # rewards /= np.std(rewards) + 1e-8
        # rewards -= np.mean(rewards)
        discounted = []
        r = 0.0
        for i in reversed(range(len(rewards))):
            r = rewards[i] + args.gamma * r
            discounted.append(r)

        discounted = list(reversed(discounted))
        discounted = np.array(discounted)
        # discounted /= np.std(discounted) + 1e-8
        # print (discounted)
        # input("")
        for i in range(len(discounted)):
            loss += logps[i] * Variable(torch.FloatTensor([discounted[i]]))
        total_len += len(rewards)
        # input("")
    # loss = -loss / total_len
    loss = loss / total_len
    loss.backward(retain_graph=retain)


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

def optimize(args):
    args.log_dir = os.path.join(args.log_dir, "envlqr"+"_kf"+str(int(not args.no_kalman))+\
        "_maxsamples"+str(args.max_samples)+"_batch"+str(args.batch_size)+\
        "_lr"+str(args.lr)+"_error"+str(args.kf_error_thresh)+\
        "_diag"+str(int(args.use_diagonal_approx))+"_sos"+str(args.sos_init))+\
        "_state"+str(args.x0)+"_"+str(args.y0)+"_"+str(args.xv0)+"_"+str(args.yv0)
    args.log_dir = os.path.join(args.log_dir, str(args.seed))
    os.makedirs(args.log_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = lqg.LQG_Env(state0=np.array([args.x0, args.y0, args.xv0, args.yv0]))
    # env = lcp.LinearizedCartPole()
    # K = env.K
    # env = cp.CartPoleContinuousEnv()
    pi = pth_policy.LinearPolicy(env.observation_space.shape[0], env.action_space.shape[0])
    # pi.lin.weight.data = torch.from_numpy(-K+np.random.normal(scale=3, size=K.shape)).float()
    opt = optim.Adam(pi.parameters(), lr=args.lr)

    kf = KalmanFilter(pi.num_params(),
                      use_diagonal_approx=args.use_diagonal_approx,
                      sos_init=args.sos_init,
                      reset_observation_noise=args.reset_obs_noise,
                      reset_state=args.reset_kf_state,
                      window_size=20)
    kf.reset()

    print ("Random episodes")
    rewards, _, _ = do_rollout(env, pi, eval=True)
    print ("Episode return: ", sum(rewards))
    rewards, _, _ = do_rollout(env, pi, optimal=True, eval=True)
    # print ("Optimal return over horizon", env.T,":", sum(rewards))
    print ("Optimal return over horizon:", sum(rewards))

    tgrads = []
    current_batch_samples = 0
    current_batch_trajs = 0
    total_samples = 0
    total_trajs = 0

    eval_perf = eval_pi(env, pi, avg_n=10)
    print ("Initial evaluation avg of 10: ", eval_perf, ", total samples", total_samples)

    log_batch_sizes = []
    log_est_obs_noise = []
    log_num_trajs = []
    log_num_samples = []
    log_grad_obs = []
    log_eval_perf = [eval_perf]

    batch_grads = []

    i = 0
    while total_samples < args.max_samples:
        # sample data
        rewards, glogps, states = do_rollout(env, pi)
        tgrads.append([rewards, glogps])
        current_batch_samples += len(rewards)
        total_samples += len(rewards)
        current_batch_trajs += 1
        total_trajs += 1

        if args.no_kalman:
            opt.zero_grad()
            compute_rollout_grad(args, [(rewards, glogps)], retain=True)
            grad_obs = pi.lin.weight.grad.view(-1).numpy().copy()
            batch_grads.append(grad_obs)
            log_grad_obs.append(grad_obs)

            if current_batch_samples >= args.batch_size:
                opt.zero_grad()
                compute_rollout_grad(args, tgrads)
                opt.step()

                # batch_grads = np.array(batch_grads) # grads are (ntraj x nparams)
                # print (batch_grads.shape)
                # import matplotlib.pyplot as plt
                # for i in range(batch_grads.shape[1]):
                #     print (batch_grads[:,i])
                #     plt.hist(batch_grads[:,i], bins=100, density=True)
                #     plt.show()
                batch_grads = []

                eval_perf = eval_pi(env, pi, avg_n=10)
                print ("Evaluation avg of 10: ", eval_perf, ", total samples", total_samples)

                log_batch_sizes.append(current_batch_trajs)
                log_est_obs_noise.append(np.zeros_like(kf.Rt))
                log_num_samples.append(total_samples)
                log_num_trajs.append(total_trajs)
                log_eval_perf.append(eval_perf)

                tgrads = []
                current_batch_samples = 0
                current_batch_trajs = 0

                # input("")
        else: # not args.no_kalman:
            # check for error thresh
            # print ("KF error: ", np.mean(kf.e))
            opt.zero_grad()
            compute_rollout_grad(args, [(rewards, glogps)])
            grad = pi.lin.weight.grad.view(-1, 1).numpy()
            log_grad_obs.append(grad)
            kf.update(grad)
            if np.mean(kf.e) < args.kf_error_thresh:
                print ("KF error: ", np.mean(kf.e), current_batch_trajs)
                execute_kf_grad_step(pi, kf)
                opt.step()

                eval_perf = eval_pi(env, pi, avg_n=10)
                print ("Evaluation avg of 10: ", eval_perf, ", total samples", total_samples)
                log_batch_sizes.append(current_batch_trajs)
                log_est_obs_noise.append(kf.Rt.copy())
                log_num_samples.append(total_samples)
                log_num_trajs.append(total_trajs)
                log_eval_perf.append(eval_perf)

                kf.reset()
                current_batch_samples = 0
                current_batch_trajs = 0
                tgrads = []

        i += 1

    np.save(os.path.join(args.log_dir, "log_batch_sizes.npy"), np.array(log_batch_sizes))
    np.save(os.path.join(args.log_dir, "log_est_obs_noise.npy"), np.array(log_est_obs_noise))
    np.save(os.path.join(args.log_dir, "log_num_trajs.npy"), np.array(log_num_trajs))
    np.save(os.path.join(args.log_dir, "log_num_samples.npy"), np.array(log_num_samples))
    np.save(os.path.join(args.log_dir, "log_grad_obs.npy"), np.stack(log_grad_obs))
    np.save(os.path.join(args.log_dir, "log_eval_perf.npy"), np.array(log_eval_perf))

if __name__ == '__main__':
    import arguments
    args = arguments.get_args()
    optimize(args)
