import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import funcs
import models
from rl_adaptive_sampling.opt import kalman_opt

def optimize(args):
    args.log_dir = os.path.join(args.log_dir, "batch"+str(args.batch_size)+"lr"+str(args.lr)+"error"+str(args.kf_error_thresh)+"noisyobj"+str(args.noisy_objective))
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

    f = funcs.Parabola()
    kf = kalman_opt.KalmanFilter(state_dim=2)
    model = models.GaussianModel(ndim=1)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)

    for itr in range(args.n_iters):
        kf.reset(err_init=1.0)
        log_prob_obj_loss = 0

        log_grad_est.append(kf.xt)
        log_grad_true.append(0)
        log_grad_obs.append(np.array([[0], [0]]))
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
            log_prob_obj = logprob * Variable(torch.from_numpy(objv))
            if args.no_kalman:
                log_prob_obj_loss += log_prob_obj
            else:
                opt.zero_grad()
                log_prob_obj.backward()
                grad = model.flattened_grad().numpy()
                # print ("Grad: ", grad.shape)
                # input("")
                kf.update(grad)
                if nsample > 10 and np.linalg.norm(kf.e) / kf.ndim < args.kf_error_thresh:
                    print ("Reached error: ", kf.e, kf.e.shape)
                    break

            # print ("grad est, true grad, observation: ", xt, f.jacobian(minimum), y)
            log_grad_est.append(kf.xt)
            log_grad_true.append(0)
            log_grad_obs.append(kf.y)
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
            # print (torch.from_numpy(kf.xt).float())
            model.unflatten_grad(torch.from_numpy(kf.xt).float())
        opt.step()
        print ("Approximate minimum: ", model.mu.data.numpy(), model.log_std.exp().data.numpy())
        if model.log_std.data < np.log(args.min_std):
            print ("Setting min variance to ", args.min_std)
            model.log_std.data = torch.FloatTensor([np.log(args.min_std)])

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
