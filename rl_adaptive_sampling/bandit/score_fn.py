import os

import numpy as np
import scipy.stats as ss
from scipy.integrate import quad
import funcs
import models

from rl_adaptive_sampling.opt import kalman_opt


def grad_log_normal_pdf(x, mu, sig):
    """ compute grad of log normal dist"""
    # grad w.r.t. mu
    dmu = (x-mu)/(sig**2.0)
    # grad w.r.t. sigma
    dsig = 0.0 #( np.exp(rho) * ((x-mu)**2.0 - np.log(np.exp(rho)+1)**2.0) ) / ( (np.exp(rho) + 1) * np.log(np.exp(rho) + 1)**3.0)
    # print (dmu, dsigma)
    return np.array([[float(dmu)],
                     [float(dsig)]])

def integral_func(x, model, f):
    return ss.norm.pdf(x, loc=model.mu, scale=model.std) * grad_log_normal_pdf(x, model.mu, model.std)[0,:] * f.f(x)

def optimize(args):

    args.log_dir = os.path.join(args.log_dir, "batch"+str(args.batch_size)+"_lr"+str(args.lr)+"_error"+str(args.kf_error_thresh)+"_noisyobj"+str(int(args.noisy_objective))+"_f"+args.func+"_diag"+str(int(args.use_diagonal_approx))+"_sos"+str(args.sos_init))
    args.log_dir = os.path.join(args.log_dir, str(args.seed))
    os.makedirs(args.log_dir, exist_ok=True)
    np.random.seed(args.seed)

    log_min_mu_est = []
    log_min_std_est = []
    log_grad_est = []
    log_grad_true = []
    log_grad_obs = []
    log_obs_noise_est = []
    log_cov_error = []
    log_abs_error_true = []
    log_abs_error_est = []
    log_batch_sizes = []

    f = funcs.make_func(args.func)
    model = models.GaussianModel(ndim=f.input_dimen)
    kf = kalman_opt.KalmanFilter(state_dim=model.nparam, use_diagonal_approx=args.use_diagonal_approx, use_last_error=args.use_last_error, sos_init=args.sos_init)
    true_grad = quad(integral_func, -np.inf, np.inf, args=(model, f))

    num_samples = 0
    while num_samples < args.max_samples:
        # if not args.no_kalman:
        kf.reset()
        log_prob_obj_loss = 0

        log_grad_est.append(kf.xt)
        log_grad_true.append(np.array(true_grad))
        log_grad_obs.append(np.zeros_like(kf.xt))
        log_cov_error.append(kf.Pt)
        log_min_mu_est.append(model.mu.copy())
        log_min_std_est.append(model.std.copy())
        log_abs_error_est.append(kf.e)
        log_abs_error_true.append(0) #np.max(np.abs(xt - f.jacobian(minimum))))
        log_obs_noise_est.append(kf.Rt)

        gs = []
        mu_grads = []

        for nsample in range(args.batch_size):
            z, logprob = model.sample()
            fz = f.f(z)
            if args.noisy_objective:
                fz += np.random.normal()
            # if nsample % 1000 == 0:
            #     print ("Sample: ", nsample, samp, objv)

            loggrad = grad_log_normal_pdf(z, model.mu, model.std)[0,:][:,np.newaxis]
            # loggrads.append(loggrad)
            mu_grads.append(loggrad[0])
            grad = fz * loggrad
            # print ("grad: ", grad.shape, fz.shape, loggrad.shape)
            gs.append(grad)

            if not args.no_kalman:
                kf.update(grad)
                #print (nsample, np.mean(kf.e), np.mean(kf.Rt)) #, kf.e)
                if nsample >= 1 and np.mean(kf.e) < args.kf_error_thresh:
                    # print ("Reached error: ", np.mean(kf.e))
                    # print ("Nsamples: ", nsample)
                    break

            # print ("grad est, true grad, observation: ", xt, f.jacobian(minimum), y)
            if args.no_kalman:
                gest = np.mean(np.array(gs), axis=0).copy()
                log_grad_est.append(gest)
            else:
                log_grad_est.append(kf.xt)
            # print (kf.xt.shape, grad.shape)
            # input("")

            log_grad_obs.append(grad)
            log_cov_error.append(kf.Pt)
            log_min_mu_est.append(model.mu.copy())
            log_min_std_est.append(model.std.copy())
            log_abs_error_est.append(kf.e)
            log_abs_error_true.append(0)
            log_obs_noise_est.append(kf.Rt)

        true_grad = quad(integral_func, -np.inf, np.inf, args=(model, f))
        log_grad_true.append(np.array(true_grad))

        log_batch_sizes.append(nsample + 1)
        num_samples += nsample + 1

        if args.no_kalman:
            gt = np.mean(np.array(gs), axis=0)
        else:
            gt = kf.xt

        model.mu = model.mu - args.lr * gt[0]
        # don't update std
        print ("Approximate minimum: ", model.mu, model.std)
        print ("True grad: ", true_grad)

    np.save(os.path.join(args.log_dir, "log_min_mu_est.npy"), np.array(log_min_mu_est))
    np.save(os.path.join(args.log_dir, "log_min_std_est.npy"), np.array(log_min_std_est))
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
