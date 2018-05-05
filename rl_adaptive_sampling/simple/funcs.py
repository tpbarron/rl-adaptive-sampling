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
        assert x.ndim == self.input_dimen
        y = x**2.0
        return y

    def grad(self, x):
        assert isinstance(x, np.ndarray)
        return 2.0 * x
