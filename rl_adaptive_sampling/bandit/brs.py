import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import funcs
import models
from rl_adaptive_sampling.opt import kalman_opt

def optimize(args):

    args.log_dir = os.path.join(args.log_dir, "brs_batch"+str(args.batch_size)+"_lr"+str(args.lr)+"_error"+str(args.kf_error_thresh)+"_noisyobj"+str(int(args.noisy_objective))+"_f"+args.func+"_diag"+str(int(args.use_diagonal_approx))+"_sos"+str(args.sos_init)+"_nu"+str(args.nu))
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

    f = funcs.make_func(args.func)
    model = models.SingleParameterModel(ndim=f.input_dimen)
    kf = kalman_opt.KalmanFilter(state_dim=model.nparam, error_init=1.0, sos_init=1.0, reset_state=args.reset_kf_state)
    kf.reset()

    num_samples = 0
    while num_samples < args.max_samples:
        if not args.no_kalman:
            kf.reset()

        log_grad_est.append(kf.xt)
        log_grad_true.append(0)
        log_grad_obs.append(np.zeros_like(kf.xt))
        log_cov_error.append(kf.Pt)
        log_min_est.append(model.x.copy())
        log_abs_error_est.append(kf.e)
        log_abs_error_true.append(0)
        log_obs_noise_est.append(kf.Rt)

        ys = []
        for nsample in range(args.batch_size):
            # generate random delta
            delta = np.random.normal(size=model.x.shape)
            z_minus = model.x - args.nu * delta
            z_plus = model.x + args.nu * delta

            fz_minus = f.f(z_minus)
            fz_plus = f.f(z_plus)
            if args.noisy_objective:
                fz_minus = fz_minus + np.random.normal(size=model.x.shape)
                fz_plus = fz_plus + np.random.normal(size=model.x.shape)

            grad_est = (fz_plus - fz_minus) * delta # / args.nu
            # print (grad_est, grad_est/args.nu)
            # input("")
            ys.append(grad_est)

            if not args.no_kalman:
                kf.update(grad_est)
                # print ("Current error: ", np.mean(kf.e))
                if np.mean(kf.e) < args.kf_error_thresh:
                    print ("Reached error: ", np.mean(kf.e))
                    print ("Nsamples: ", nsample)
                    break

            # print ("grad est, true grad, observation: ", xt, f.jacobian(minimum), y)
            if args.no_kalman:
                gest = np.mean(np.array(ys), axis=0).copy()
                log_grad_est.append(gest)
            else:
                log_grad_est.append(kf.xt)
            log_grad_true.append(0)
            log_grad_obs.append(grad_est)
            log_cov_error.append(kf.Pt)
            log_min_est.append(model.x.copy())
            log_abs_error_est.append(kf.e)
            log_abs_error_true.append(0)
            log_obs_noise_est.append(kf.Rt)

        if args.no_kalman:
            gt = np.mean(np.array(ys), axis=0)
        else:
            gt = kf.xt
        model.x = model.x - args.lr * gt
        print ("Approximate minimum: ", model.x)
        log_batch_sizes.append(nsample+1)
        num_samples += nsample+1

    print ("total samples: ", sum(log_batch_sizes))

    # print (log_grad_obs[0:10])
    # for i in range(len(log_grad_obs)):
    #     x = log_grad_obs[i]
    #     print (i, x)
    #     if x.shape != kf.xt.shape:
    #         print (x.shape)

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
