import numpy as np

class Function(object):

    def __init__(self, input_dimen=1, output_dimen=1):
        self.input_dimen = input_dimen
        self.output_dimen = output_dimen

    def f(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

class Parabola(Function):
    """
    y = x^2 + N(0,1)*sigma
    Jacobian: dy/dx = 2x
    Hessian: d^2y/d^2x = 2
    """
    def __init__(self):
        super(Parabola, self).__init__(input_dimen=1, output_dimen=1)

    def f(self, x):
        assert x.size == self.input_dimen
        y = x**2.0
        return y

    def grad(self, x):
        assert isinstance(x, np.ndarray)
        return 2.0 * x

class NDQuadratic(Function):
    """
    Test case for high-D function when gradients are HIGHLY correlated
    """
    def __init__(self, ndim=10, Q=None):
        super(NDQuadratic, self).__init__(input_dimen=ndim, output_dimen=1)
        self.ndim = ndim
        if Q is None:
            Q = np.eye(self.ndim)
        self.Q = Q

    def f(self, x):
        assert x.size == self.input_dimen
        if x.ndim == 1:
            x = x[:,np.newaxis]
        y = np.transpose(x) @ self.Q @ x
        return y

    def grad(self, x):
        j = 2 * self.Q @ x
        return j


class Quartic(Function):
    def __init__(self):
        super(Quartic, self).__init__(input_dimen=1, output_dimen=1)

    def f(self, x):
        assert x.ndim == self.input_dimen
        y = 2 * x**4.0 + 2 * x**3
        return y

    def grad(self, x):
        raise NotImplementedError


from scipy.optimize import rosen, rosen_der

class Rosenbrock(Function):

    def __init__(self):
        super(Rosenbrock, self).__init__(input_dimen=2, output_dimen=1)

    def f(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array([x])
        y = rosen(x)
        return np.array([y])

    def grad(self, x):
        g = rosen_der(x)
        return g


def make_func(fname):
    if fname == 'parabola':
        return Parabola()
    if fname == 'ndquad':
        return NDQuadratic()
    if fname == 'rosen':
        return Rosenbrock()
    if fname =='quartic':
        return Quartic()

if __name__ == '__main__':
    f = Rosenbrock()
    y = f.f(np.array([1, 1]))
    print (y)
    g = f.grad(np.array([1, 1]))
    print (g)

    y = f.f(np.array([0, 1]))
    print (y)
    g = f.grad(np.array([0, 1]))
    print (g)
