import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import funcs
import models

from rl_adaptive_sampling.opt import kalman_opt

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
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)

    for itr in range(args.n_iters):
        if not args.no_kalman:
            kf.reset()
        log_prob_obj_loss = 0

        log_grad_est.append(kf.xt)
        log_grad_true.append(0)
        log_grad_obs.append(np.zeros_like(kf.xt))
        log_cov_error.append(kf.Pt)
        log_min_mu_est.append(model.mu.data.numpy())
        log_min_std_est.append(model.log_std.exp().data.numpy())
        log_abs_error_est.append(kf.e)
        log_abs_error_true.append(0) #np.max(np.abs(xt - f.jacobian(minimum))))
        log_obs_noise_est.append(kf.Rt)

        for nsample in range(args.batch_size):
            samp, logprob = model()
            objv = f.f(samp.data.numpy())
            if args.noisy_objective:
                objv += np.random.normal()
            # if nsample % 1000 == 0:
            #     print ("Sample: ", nsample, samp, objv)
            log_prob_obj = logprob * Variable(torch.from_numpy(objv).float())
            grad = log_grad_obs[0] # default zero'd
            if args.no_kalman:
                log_prob_obj_loss += log_prob_obj
            else:
                opt.zero_grad()
                log_prob_obj.backward()
                grad = model.flattened_grad().numpy()
                #print ("Grad: ", grad, grad.shape)
                # input("")
                kf.update(grad)
                #print (nsample, np.mean(kf.e), np.mean(kf.Rt)) #, kf.e)
                if nsample >= 10 and np.mean(kf.e) < args.kf_error_thresh:
                    #print ("Reached error: ", np.mean(kf.e))
                    #print ("Nsamples: ", nsample)
                    break
                # if nsample >= 100 and np.linalg.norm(kf.e)**2.0/kf.state_dim < args.kf_error_thresh:
                #     print ("Reached error: ", np.linalg.norm(kf.e)**2.0/kf.state_dim) #, kf.e.shape)
                #     print ("Nsamples: ", nsample)
                #     break

            # print ("grad est, true grad, observation: ", xt, f.jacobian(minimum), y)
            log_grad_est.append(kf.xt)
            log_grad_true.append(0)
            log_grad_obs.append(grad)
            log_cov_error.append(kf.Pt)
            log_min_mu_est.append(model.mu.data.numpy().copy())
            log_min_std_est.append(model.log_std.exp().data.numpy().copy())
            log_abs_error_est.append(kf.e)
            log_abs_error_true.append(0)
            log_obs_noise_est.append(kf.Rt)

        log_batch_sizes.append(nsample)
        opt.zero_grad()
        if args.no_kalman:
            (log_prob_obj_loss/args.batch_size).backward()
        else:
            model.unflatten_grad(torch.from_numpy(kf.xt).float())
        opt.step()
        #print ("Approximate minimum: ", model.mu.data.numpy(), model.log_std.exp().data.numpy())
        if np.any(model.log_std.data.numpy() < np.log(args.min_std)):
            #print ("Setting min variance to ", args.min_std)
            for p in range(model.nparam//2):
                model.log_std.data[p] = np.log(args.min_std)

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
