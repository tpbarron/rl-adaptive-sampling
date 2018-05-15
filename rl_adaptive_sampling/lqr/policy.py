import numpy as np
import scipy.stats as ss


class LinearPolicy(object):

    def __init__(self, input_dim, action_dim):
        print ("Linear policy: ", input_dim, action_dim)
        self.W = np.random.normal(size=(action_dim, input_dim+1)) * 0.1
        self.std = np.ones((action_dim,)) * 0.1

        self.alpha = 0.0005
        self.beta_1 = 0.9
        self.beta_2 = 0.999						#initialize the values of the parameters
        self.epsilon = 1e-8

        self.m_t = 0
        self.v_t = 0
        self.t = 0


    def grad_log_normal_pdf(self, x, mu, sig):
        """ compute grad of log normal dist wrt mu """
        # grad w.r.t. mu
        dmu = (x-mu)/(sig**2.0)
        # grad w.r.t. sigma
        # dsig = 0.0 #( np.exp(rho) * ((x-mu)**2.0 - np.log(np.exp(rho)+1)**2.0) ) / ( (np.exp(rho) + 1) * np.log(np.exp(rho) + 1)**3.0)
        # print (dmu, dsigma)
        return dmu #np.array([[float(dmu)],
                        #  [float(dsig)]])

    def act(self, state):
        state = np.concatenate((state, np.array([1])))
        mu = np.dot(self.W, state)
        a = np.random.normal(mu, self.std)
        # logp = ss.norm.logpdf(a, loc=mu, scale=self.std)
        # print (logp.shape, self.W.shape, state.shape)
        glogp = self.grad_log_normal_pdf(a, mu, self.std)
        glogpw = np.dot(glogp[:,np.newaxis], np.transpose(state[:,np.newaxis])).copy()
        # print (glogpw.shape)
        # input("")
        return a, glogpw


    def apply(self, g_t):
        self.t+=1
        self.m_t = self.beta_1*self.m_t + (1-self.beta_1)*g_t	#updates the moving averages of the gradient
        self.v_t = self.beta_2*self.v_t + (1-self.beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
        m_cap = self.m_t/(1-(self.beta_1**self.t))		#calculates the bias-corrected estimates
        v_cap = self.v_t/(1-(self.beta_2**self.t)) #calculates the bias-corrected estimates
        self.W = self.W - (self.alpha*m_cap)/(np.sqrt(v_cap)+self.epsilon)

        # self.W = self.W - self.alpha * g_t
