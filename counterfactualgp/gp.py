import autograd
import autograd.numpy as np
import sys

from autograd.scipy.misc import logsumexp
from numpy.linalg.linalg import LinAlgError
from scipy.optimize import minimize

from counterfactualgp.autodiff import packing_funcs, vec_mvn_logpdf


class GP:
    def __init__(self, mean_fn, cov_fn, tr_fns=[], ac_fn=None):
        self.mean = mean_fn
        self.cov = cov_fn

        self.params = {}
        self.params.update(self.mean(params_only=True))
        self.params.update(self.cov(params_only=True))

        if tr_fns:
            self.tr = tr_fns
            for tr in self.tr:
                self.params.update(tr(params_only=True))
        self.tr = [lambda *args, **kwargs: 0] + self.tr

        if ac_fn:
            self.action = ac_fn
            self.params.update(self.action(params_only=True))
        else:
            self.action = lambda *args, **kwargs: [1.0]

    def predict(self, x_star, y, x):
        p_a = self.action(self.params)
        
        l = len(x_star[0])
        m = np.zeros(l)
        c = np.zeros([l, l])
        for _p_a, tr in zip(p_a, self.tr):
            _m, _c = self._predict(x_star, y, x, treatment=tr)
            m += _p_a * _m
            c = _c # All covariance matrix are the same

        return m, c

    def _predict(self, x_star, y, x, treatment):
        t_star, rx_star = x_star
        prior_mean = self.mean(self.params, t_star)
        prior_mean += treatment(self.params, x_star, prior_mean)
        prior_cov = self.cov(self.params, t_star)

        if len(y) == 0:
            return prior_mean, prior_cov

        t, rx = x
        obs_mean = self.mean(self.params, t)
        obs_mean += treatment(self.params, x, obs_mean)
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
        fixed_params = dict([(k,v) for k,v in self.params.items() if k.endswith('_F')])
        pack, unpack = packing_funcs(trainable_params)

        def obj(w):
            p = unpack(w)
            p.update(fixed_params)
            f = 0.0

            ln_p_a = np.log(self.action(p)) # individual- and time-invariant
            
            for y, x in samples:
                # Outcome model
                mixture = [_ln_p_a + log_likelihood(p, y, x, mean_fn=self.mean, cov_fn=self.cov, tr_fn=tr) 
                        for _ln_p_a, tr in zip(ln_p_a, self.tr)]
                f -= logsumexp(np.array(mixture))

                # Action model
                _, rx = x
                n_rx = [np.sum(rx == i) for i in range(ln_p_a.shape[0])]
                f -= np.dot(ln_p_a.T, np.array(n_rx))

            # Regularizers
            for k,_ in trainable_params.items():
                if k.endswith('_F'):
                    f += np.sum(p[k]**2)

            return f

        def callback(w):
            print('obj=', obj(w))

        grad = autograd.grad(obj)

        solution = minimize(obj, pack(self.params), jac=grad, method='BFGS', callback=callback)
        self.params = unpack(solution['x'])
        self.params.update(fixed_params)

    def _initialize(self, samples):
        self.params = self.mean(self.params, samples, params_only=True)


def log_likelihood(params, y, x, mean_fn, cov_fn, tr_fn):
    t, rx = x
    m = mean_fn(params, t)
    m += tr_fn(params, x, m)
    c = cov_fn(params, t)

    ln_p_y = vec_mvn_logpdf(y, m, c)

    return ln_p_y
