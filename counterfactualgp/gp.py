import autograd
import autograd.numpy as np

from numpy.linalg.linalg import LinAlgError
from scipy.optimize import minimize

from counterfactualgp.autodiff import packing_funcs, vec_mvn_logpdf
from counterfactualgp import mean


class GP:
    def __init__(self, degree):
        self.degree = degree
        self.params = {}
        self.params.update(mean.linear_params(degree))
        self.params['ln_cov_y'] = np.zeros(1)

    def predict(self, x_star, y, x):
        t_star, rx_star = x_star
        prior_mean = mean.linear_predict(self.params, t_star)
        prior_cov = cov_fn(self.params, t_star)

        if len(y) == 0:
            return prior_mean, prior_cov

        t, rx = x
        obs_mean = mean.linear_predict(self.params, t)
        obs_cov = cov_fn(self.params, t)

        cross_cov = cov_fn(self.params, t_star, t)

        alpha = np.linalg.solve(obs_cov, cross_cov.T).T
        mean = prior_mean + np.dot(alpha, y - obs_mean)
        cov = prior_cov - np.dot(alpha, cross_cov.T)

        return mean, cov

    def fit(self, samples, init = True):
        if init:
            self._initialize(samples)

        pack, unpack = packing_funcs(self.params)

        def obj(w):
            p = unpack(w)
            f = 0.0

            for y, x in samples:
                f -= log_likelihood(p, y, x)

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
        p = mean.linear_fit_params(self.params, samples)
        self.params.update(p)


def log_likelihood(params, y, x):
    t, rx = x
    m = mean.linear_predict(params, t)
    c = cov_fn(params, t)

    ln_p_y = vec_mvn_logpdf(y, m, c)

    return ln_p_y


def cov_fn(params, t1, t2=None, eps=1e-3):
    v_y = np.exp(params['ln_cov_y'])
    symmetric = t2 is None

    if symmetric:
        cov = (v_y + eps) * np.eye(len(t1))
    else:
        cov = np.zeros((len(t1), len(t2)))

    return cov
