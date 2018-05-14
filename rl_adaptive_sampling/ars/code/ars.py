'''
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht
'''

import parser
import time
import os
import numpy as np
import gym
import logz
import optimizers
from policies import *
import kalman_opt
from rl_adaptive_sampling.opt import kalman_opt
from collections import deque

class Worker(object):
    """
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 env_name='',
                 policy_params = None,
                 rollout_length=1000,
                 delta_std=0.02):

        # initialize OpenAI environment for each worker
        self.env = gym.make(env_name)
        self.env.seed(env_seed)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table.
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        else:
            raise NotImplementedError

        self.delta_std = delta_std
        self.rollout_length = rollout_length


    def get_weights_plus_stats(self):
        """
        Get current policy weights and current statistics of past states.
        """
        assert self.policy_params['type'] == 'linear'
        return self.policy.get_weights_plus_stats()


    def rollout(self, shift = 0., rollout_length = None):
        """
        Performs one rollout of maximum length rollout_length.
        At each time-step it substracts shift from the reward.
        """

        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0

        ob = self.env.reset()
        for i in range(rollout_length):
            action = self.policy.act(ob)
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                break

        return total_reward, steps

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        """
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas = [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas.append(-1)

                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
                rollout_rewards.append(reward)

            else:
                # idx, delta = self.deltas.get_delta(w_policy.size)
                delta_i = np.random.randn(w_policy.size)

                delta = (self.delta_std * delta_i).reshape(w_policy.shape)
                deltas.append(delta_i.copy())

                # set to true so that state statistics are updated
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta.copy())
                pos_reward, pos_steps  = self.rollout(shift = shift)

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta.copy())
                neg_reward, neg_steps = self.rollout(shift = shift)
                steps += pos_steps + neg_steps

                rollout_rewards.append([pos_reward, neg_reward])

        return {'deltas': deltas, 'rollout_rewards': rollout_rewards, "steps" : steps}

    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()


class ARSLearner(object):
    """
    Object class implementing the ARS algorithm.
    """

    def __init__(self, env_name='HalfCheetah-v1',
                 policy_params=None,
                 num_workers=32,
                 num_deltas=320,
                 deltas_used=320,
                 delta_std=0.02,
                 logdir=None,
                 rollout_length=1000,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 use_kf=True,
                 kf_error_thresh=0.1,
                 kf_sos=0.1,
                 seed=123):

        logz.configure_output_dir(logdir)
        logz.save_params(params)

        env = gym.make(env_name)

        self.timesteps = 0
        self.episodes = 0
        self.action_size = env.action_space.shape[0]
        self.ob_size = env.observation_space.shape[0]
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')
        self.use_kf = use_kf
        self.kf_error_thresh = kf_error_thresh
        self.kf_sos = kf_sos

        # initialize workers with different random seeds
        print('Initializing workers.')
        self.num_workers = num_workers
        self.worker = Worker(seed + 7,
                             env_name=env_name,
                             policy_params=policy_params,
                             rollout_length=rollout_length,
                             delta_std=delta_std)

        # initialize policy
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError

        if self.use_kf:
            self.kf = kalman_opt.KalmanFilter(self.w_policy.size, sos_init=self.kf_sos, reset_state=True, reset_observation_noise=False)
            self.reward_std = deque(maxlen=1000)
            self.reward_std.append(1)

        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)
        print("Initialization of ARS complete.")

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts

        t1 = time.time()
        num_rollouts = self.num_deltas # int(num_deltas / self.num_workers)

        results = self.worker.do_rollouts(self.w_policy, num_rollouts=num_rollouts, shift=self.shift, evaluate=evaluate)

        # print (results)
        if not evaluate:
            self.timesteps += results["steps"]
            self.episodes += 2 * num_rollouts
        deltas = np.array(results['deltas'])
        rollout_rewards = np.array(results['rollout_rewards'], dtype=np.float64)

        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas

        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        deltas = deltas[idx]
        rollout_rewards = rollout_rewards[idx,:]

        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards) + 1e-8

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step

        import utils
        deltas_list = [deltas[i].flatten().copy() for i in range(len(deltas))]
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  deltas_list,
                                                  batch_size = 500)

        g_hat /= num_deltas #len(weights)
        # print ("num deltas: ", num_deltas, num_rollouts, deltas.size)
        # input("")
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat.copy()


    def aggregate_rollouts_kf(self, num_rollouts = None, evaluate = False):
        """
        Aggregate update step from rollouts generated in parallel.
        """
        max_rollouts = self.num_deltas * 1000

        t1 = time.time()

        itraj = 0
        g_hat = 0.0

        self.kf.reset()

        while itraj < max_rollouts and np.mean(self.kf.e) > self.kf_error_thresh:
            results = self.worker.do_rollouts(self.w_policy, num_rollouts=1, shift=self.shift, evaluate=evaluate)

            rollout_rewards, deltas = results['rollout_rewards'], results['deltas']
            if not evaluate:
                self.episodes += 2
                self.timesteps += results["steps"]

            deltas = np.array(deltas)
            rollout_rewards = np.array(rollout_rewards, dtype = np.float64)

            if evaluate:
                return rollout_rewards

            # print (rollout_rewards)
            for r in rollout_rewards.flatten().tolist():
                self.reward_std.append(r)
            # normalize rewards by their standard deviation
            rollout_rewards /= np.std(np.asarray(self.reward_std)) # np.std(rollout_rewards)

            weights = rollout_rewards[:,0] - rollout_rewards[:,1]
            g_hat_i = np.dot(weights[0], deltas[0])
            self.kf.update(g_hat_i.flatten()[:,np.newaxis])

            g_hat /= deltas.size

            itraj += 1

        print ("Kalman error: ", np.mean(self.kf.e), itraj)
        return self.kf.xt


    def train_step(self):
        """
        Perform one update step of the policy weights.
        """

        if self.use_kf:
            g_hat = self.aggregate_rollouts_kf()
        else:
            g_hat = self.aggregate_rollouts()
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        # print ("Ghat: ", g_hat)
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        # print ("Close: ", np.allclose(self.w_policy, self.w_policy_old))
        return

    def train(self, num_iter):

        start = time.time()

        i = 0
        while self.episodes < num_iter:
        # for i in range(num_iter):
            t1 = time.time()
            self.train_step()
            t2 = time.time()
            print('total time of one step', t2 - t1)
            print('iter ', i,' done')

            # record statistics every 10 iterations
            if ((i + 1) % 10 == 0):

                rewards = self.aggregate_rollouts(num_rollouts = 100, evaluate = True)
                w = self.worker.get_weights_plus_stats()
                np.savez(self.logdir + "/lin_policy_plus", w)

                print(sorted(self.params.items()))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("AverageReward", np.mean(rewards))
                logz.log_tabular("StdRewards", np.std(rewards))
                logz.log_tabular("MaxRewardRollout", np.max(rewards))
                logz.log_tabular("MinRewardRollout", np.min(rewards))
                logz.log_tabular("timesteps", self.timesteps)
                logz.log_tabular("episodes", self.episodes)
                logz.dump_tabular()

            i += 1

        return

def run_ars(params):

    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    env = gym.make(params['env_name'])
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type':'linear',
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim}

    ARS = ARSLearner(env_name=params['env_name'],
                     policy_params=policy_params,
                     num_workers=params['n_workers'],
                     num_deltas=params['n_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'],
                     logdir=logdir,
                     rollout_length=params['rollout_length'],
                     shift=params['shift'],
                     params=params,
                     seed = params['seed'],
                     use_kf=params['use_kf'],
                     kf_error_thresh=params['kf_error_thresh'],
                     kf_sos=params['kf_sos'])

    ARS.train(params['n_iter'])

    return


from envs.cartpole import CartPoleContinuousEnv
from gym.envs.registration import register

register(
    id='CartPoleContinuous-v0',
    entry_point='envs.cartpole:CartPoleContinuousEnv',
)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPoleContinuous-v0')
    parser.add_argument('--n_iter', '-n', type=int, default=250)
    parser.add_argument('--n_directions', '-nd', type=int, default=1)
    parser.add_argument('--deltas_used', '-du', type=int, default=1)
    parser.add_argument('--step_size', '-s', type=float, default=0.1)
    parser.add_argument('--delta_std', '-std', type=float, default=.1)
    parser.add_argument('--n_workers', '-e', type=int, default=18)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')

    parser.add_argument('--use_kf', action='store_true', default=False)
    parser.add_argument('--kf_error_thresh', type=float, default=0.1)
    parser.add_argument('--kf_sos', type=float, default=0.1)

    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')

    # local_ip = socket.gethostbyname(socket.gethostname())
    # ray.init() #redis_address= local_ip + ':6379')

    args = parser.parse_args()
    params = vars(args)
    run_ars(params)
