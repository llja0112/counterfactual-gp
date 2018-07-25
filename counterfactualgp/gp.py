import autograd
import autograd.numpy as np

from numpy.linalg.linalg import LinAlgError
from scipy.optimize import minimize

from counterfactualgp.autodiff import packing_funcs, vec_mvn_logpdf


class GP:
    def __init__(self, mean_fn, cov_fn):
        self.mean = mean_fn
        self.cov = cov_fn
        self.params = {}
        self.params.update(self.mean(params_only=True))
        self.params.update(self.cov(params_only=True))

    def predict(self, x_star, y, x):
        t_star, rx_star = x_star
        prior_mean = self.mean(self.params, t_star)
        prior_cov = self.cov(self.params, t_star)

        if len(y) == 0:
            return prior_mean, prior_cov

        t, rx = x
        obs_mean = self.mean(self.params, t)
        obs_cov = self.cov(self.params, t)

        cross_cov = self.cov(self.params, t_star, t)

        alpha = np.linalg.solve(obs_cov, cross_cov.T).T
        mean = prior_mean + np.dot(alpha, y - obs_mean)
        cov = prior_cov - np.dot(alpha, cross_cov.T)

        return mean, cov

    def fit(self, samples, init = True):
        if init:
            self._initialize(samples)

        trainable_params = dict([(k,v) for k,v in self.params.items() if not k.endswith('_F')])
        pack, unpack = packing_funcs(trainable_params)

        def obj(w):
            p = unpack(w)
            f = 0.0

            for y, x in samples:
                f -= log_likelihood(p, y, x, mean_fn=self.mean, cov_fn=self.cov)

            for k,v in p.items():
                if k.endswith('_F'):
                    f += np.sum(v**2)

            return f

        def callback(w):
            print('obj=', obj(w))

        grad = autograd.grad(obj)

        solution = minimize(obj, pack(self.params), jac=grad, method='BFGS', callback=callback)
        self.params = unpack(solution['x'])

    def _initialize(self, samples):
        self.params = self.mean(self.params, samples, params_only=True)


def log_likelihood(params, y, x, mean_fn, cov_fn):
    t, rx = x
    m = mean_fn(params, t)
    c = cov_fn(params, t)

    ln_p_y = vec_mvn_logpdf(y, m, c)

    return ln_p_y
