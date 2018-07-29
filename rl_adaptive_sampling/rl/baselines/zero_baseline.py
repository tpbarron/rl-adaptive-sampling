# From: https://github.com/aravindr93/mjrl/blob/master/mjrl/baselines/zero_baseline.py
import numpy as np
import copy

class ZeroBaseline:
    def __init__(self, env, **kwargs):
        n = env.observation_space.shape[0]  # number of states
        self._coeffs = None

    def fit(self, paths, return_errors=False):
        if return_errors:
            return 1.0, 1.0

    def predict(self, state):
        return 0.0
