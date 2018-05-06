import sys
import numpy as np
import torch
from torch.optim.optimizer import Optimizer

class NaturalSGD(Optimizer):
    """
    """
    def __init__(self,
                 parameters,
                 lr=0.05):
        defaults = dict(lr=lr)
        super(NaturalSGD, self).__init__(parameters, defaults)

        # just get the total params in the model
        self.numel = 0
        for p in self.param_groups[0]['params']:
            self.numel += p.numel()

        # create storage for Fisher of params
        self.F = torch.FloatTensor(self.numel, self.numel).zero_()
        self.n = 0

    def get_flattened_grad(self):
        """
        Take the gradients in the .grad attribute of each param and flatten it into a vector
        """
        g = torch.FloatTensor(self.numel)
        ind = 0
        for p in self.param_groups[0]['params']:
            gnumel = p.grad.data.numel()
            g[ind:ind+gnumel] = p.grad.data.view(-1)
            ind += gnumel
        # reshape to nx1 vector
        g = g.view(-1, 1)
        return g

    def unflatten_grad(self, grad):
        # take gradient and reshape back into param matrices
        ps = []
        i = 0
        for p in self.param_groups[0]['params']:
            sh = p.shape
            s = i
            e = i + p.numel()
            # get next bunch and make proper shape
            ps.append(grad[s:e].view(p.shape))
            i = e
        return ps

    # def update_fisher(self, action_log_prob, alpha=0.9):
    #     """
    #     Updating the fisher online.
    #     """
    #     Fi = torch.FloatTensor(self.F.shape).zero_()
    #     self.n += 1
    #
    #     self.zero_grad()
    #     action_log_prob.backward(retain_graph=True)
    #     grad = self.get_flattened_grad()
    #     Fi = torch.mm(grad, torch.transpose(grad, 0, 1))
    #     self.F = alpha * self.F + (1 - alpha) * Fi
    #     min_eig = torch.min(torch.eig(self.F)[0])
    #     while min_eig < 0:
    #         print ("Fisher eig < 0: ", float(min_eig))
    #         self.F = self.F + 1e-8 * torch.eye(self.F.shape[0])
    #         min_eig = torch.min(torch.eig(self.F)[0])

    def compute_fisher(self, action_log_probs):
        # compute logprob grad for each action
        grad_log_probs = []
        self.F.zero_()
        for i in range(len(action_log_probs)):
            # For each log pi(a | s) get the gradient wrt theta
            self.zero_grad()
            action_log_probs[i].backward(retain_graph=True)
            grad = self.get_flattened_grad()
            grad_log_probs.append(grad)
        # Build matrix of gradients (nparams x nsamples)
        X = torch.cat(grad_log_probs, dim=1)
        self.F = torch.mm(X, torch.transpose(X, 0, 1))
        self.F = self.F / float(len(action_log_probs))
        # check if Fisher PSD, if not add eps * eye
        min_eig = torch.min(torch.eig(self.F)[0])
        trys = 0
        while min_eig < 0:
            trys += 1
            print ("Fisher eig < 0: ", float(min_eig))
            self.F = self.F + 0.1 * torch.eye(self.F.shape[0])
            min_eig = torch.min(torch.eig(self.F)[0])
            if trys > 100:
                print ("Fisher numerically unstable, ND.")
                sys.exit()

    def pinv_svd(self, M):
        # Compute SVD pinv.
        u, s, v = torch.svd(M)
        Minv = torch.mm(torch.mm(v, torch.diag(1/s)), u.t())
        return Minv

    def step(self, closure=None):
        """
        """
        # compute grad
        euclidean_grad = self.get_flattened_grad()
        Finv = self.pinv_svd(self.F)
        riemann_grad = torch.mm(Finv, euclidean_grad)
        # compute step size in probability space
        alpha = float(torch.sqrt(self.param_groups[0]['lr'] /
            (torch.mm(torch.mm(torch.transpose(euclidean_grad, 0, 1), Finv), euclidean_grad))))

        riemann_grads_unflattened = self.unflatten_grad(riemann_grad)

        if np.isnan(alpha):
            print (alpha, euclidean_grad, Finv, torch.mm(torch.mm(torch.transpose(euclidean_grad, 0, 1), Finv), euclidean_grad))
            np.save("Finv.npy", Finv.numpy())
            sys.exit()
        if np.any(np.isnan(self.F.numpy())) or np.any(np.isnan(Finv)):
            print (self.F)
            print (Finv)
            sys.exit()

            # input("")
        # now just do sgd, param = - alpha * param
        i = 0
        for p in self.param_groups[0]['params']:
            p.data.add_(-alpha, riemann_grads_unflattened[i])
            i += 1

        return None


def pinv_svd(M):
    # Compute SVD pinv.
    u, s, v = torch.svd(M)
    Minv = torch.mm(torch.mm(v, torch.diag(1/s)), u.t())
    return Minv

if __name__ == "__main__":
    M = torch.FloatTensor(3, 3).normal_()
    Minv = torch.inverse(M)
    Mpinv = pinv_svd(M)
    print (torch.dist(Minv, Mpinv))
