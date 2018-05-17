import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

sns.set()
sns.set_style("white")
sns.set_context("paper")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='x-small')
# plt.rc('ytick', labelsize='x-small')
fig = plt.figure(figsize=(4, 3))

x = np.linspace(-6, 6, 1000)
y = 1/2 * x ** 2.0
plt.plot(x, y, label=r'$y = \frac{1}{2} x^2$', ls='solid')

# tangent at x = 2
# grad = 2 * x - 2
grad = 1 * x - 1/2
plt.plot(x, grad, label=r'$\frac{\partial f(1)}{\partial x}$', ls='dashed')

# grad distribution
grad_dist = ss.norm.pdf(x, 1, 1)
plt.plot(x, grad_dist, label=r'pdf of $\frac{\partial f(1)}{\partial x}$', ls='solid')

plt.xlim((-4, 4))
plt.ylim((0, 1.5))
plt.xticks(np.arange(-4, 5, step=2))
plt.yticks(np.arange(0, 2, step=0.5))

plt.legend(loc="upper left")
plt.tight_layout()
# sns.despine()

# plt.show()
plt.savefig(fname='intuition.pdf', format='pdf')
