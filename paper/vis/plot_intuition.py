import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(4, 3))

x = np.linspace(-8, 8, 1000)
y = 1/2 * x ** 2.0
plt.plot(x, y, label=r'$y = 1/2  x^2$', ls='solid')

# tangent at x = 2
grad = 2 * x - 2
plt.plot(x, grad, label=r'$\frac{\partial f(2)}{\partial x}$', ls='dashed')

# grad distribution
grad_dist = ss.norm.pdf(x, 2, 1)
plt.plot(x, grad_dist, label=r'pdf of $\frac{\partial f(2)}{\partial x}$', ls='solid')

plt.xlim((-8, 8))
plt.ylim((0, 10))

plt.legend(loc="upper left")
plt.tight_layout()
# plt.show()
plt.savefig(fname='intuition.pdf', format='pdf')
