import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import funcs
import models
from rl_adaptive_sampling.opt import kalman_opt

def optimize(args):
    args.log_dir = os.path.join(args.log_dir, "batch"+str(args.batch_size)+"lr"+str(args.lr)+"error"+str(args.kf_error_thresh)+"nu"+str(args.nu))
    args.log_dir = os.path.join(args.log_dir, str(args.seed))
    os.makedirs(args.log_dir, exist_ok=True)
    np.random.seed(args.seed)

    log_min_est = []
    log_grad_est = []
    log_grad_true = []
    log_grad_obs = []
    log_obs_noise_est = []
    log_cov_error = []
    log_abs_error_true = []
    log_abs_error_est = []
    log_batch_sizes = []

    f = funcs.Parabola()
    kf = kalman_opt.KalmanFilter(state_dim=1)
    model = models.SingleParameterModel(ndim=1)

    for itr in range(args.n_iters):
        kf.reset(err_init=1.0)

        log_grad_est.append(kf.xt)
        log_grad_true.append(0)
        log_grad_obs.append(np.array([0]))
        log_cov_error.append(kf.Pt)
        log_min_est.append(model.x)
        log_abs_error_est.append(kf.e)
        log_abs_error_true.append(0)
        log_obs_noise_est.append(kf.Rt)

        ys = []
        for nsample in range(args.batch_size):
            # generate random delta
            delta = np.random.normal()
            z_minus = model.x - args.nu * delta
            z_plus = model.x + args.nu * delta

            fz_minus = f.f(z_minus[0])
            fz_plus = f.f(z_plus[0])
            if args.noisy_objective:
                fz_minus = fz_minus + np.random.normal()
                fz_plus = fz_plus + np.random.normal()

            grad_est = (fz_plus - fz_minus) / args.nu
            # print (grad_est.shape)
            ys.append(grad_est)

            if not args.no_kalman:
                # already know grad just modulate by objective
                kf.update(grad_est)
                if np.linalg.norm(kf.e) / kf.ndim < args.kf_error_thresh:
                    print ("Reached error: ", np.linalg.norm(kf.e)) #, kf.e.shape)
                    print ("Nsamples: ", nsample)
                    break

            # print ("grad est, true grad, observation: ", xt, f.jacobian(minimum), y)
            log_grad_est.append(kf.xt)
            log_grad_true.append(0)
            log_grad_obs.append(grad_est)
            log_cov_error.append(kf.Pt)
            log_min_est.append(model.x)
            log_abs_error_est.append(kf.e)
            log_abs_error_true.append(0)
            log_obs_noise_est.append(kf.Rt)

        if args.no_kalman:
            gt = np.mean(np.array(ys), axis=0)
        else:
            gt = kf.xt
        model.x = model.x - args.lr * gt
        print ("Approximate minimum: ", model.x)

    np.save(os.path.join(args.log_dir, "log_min_est.npy"), np.array(log_min_est))
    np.save(os.path.join(args.log_dir, "log_grad_est.npy"), np.array(log_grad_est))
    np.save(os.path.join(args.log_dir, "log_grad_true.npy"), np.array(log_grad_true))
    np.save(os.path.join(args.log_dir, "log_grad_obs.npy"), np.stack(log_grad_obs))
    np.save(os.path.join(args.log_dir, "log_cov_error.npy"), np.array(log_cov_error))
    np.save(os.path.join(args.log_dir, "log_abs_error_true.npy"), np.array(log_abs_error_true))
    np.save(os.path.join(args.log_dir, "log_abs_error_est.npy"), np.array(log_abs_error_est))
    np.save(os.path.join(args.log_dir, "log_obs_noise_est.npy"), np.array(log_obs_noise_est))
    np.save(os.path.join(args.log_dir, "log_batch_sizes.npy"), np.array(log_batch_sizes))

if __name__ == "__main__":
    import arguments
    optimize(arguments.get_args())
