import autograd
import autograd.numpy as np
import sys

from autograd.scipy.misc import logsumexp
from numpy.linalg.linalg import LinAlgError
from scipy.optimize import minimize

from counterfactualgp.autodiff import packing_funcs, vec_mvn_logpdf


class GP:
    def __init__(self, mean_fn, cov_fn, tr_fn=None, counterfactual=True):
        self.mean = mean_fn
        self.cov = cov_fn

        self.params = {}
        self.params.update(self.mean(params_only=True))
        self.params.update(self.cov(params_only=True))

        if tr_fn:
            self.tr = tr_fn
            self.params.update(self.tr(params_only=True))
        else:
            self.tr = lambda *args, **kwargs: 0

        if counterfactual:
            self.cf_param_key = 'action'
            self.params[self.cf_param_key] = np.zeros(1)
        else:
            self.cf_param_key = 'action_F'
            self.params[self.cf_param_key] = sys.float_info.max

    def predict(self, x_star, y, x):
        rx_logit = self.params[self.cf_param_key]
        p_rx = 1.0 / (1.0 + np.exp(-rx_logit)) # [0.5, 1)
        
        l = len(x_star[0])
        m = np.zeros(l)
        c = np.zeros([l, l])
        for _p_rx, tr in zip(p_rx, [True, None]):
            _m, _c = self._predict(x_star, y, x, treatment=tr)
            m += _p_rx * _m
            c += _p_rx * _c

        return m, c

    def _predict(self, x_star, y, x, treatment=True):
        t_star, rx_star = x_star
        prior_mean = self.mean(self.params, t_star)
        if treatment:
            prior_mean += self.tr(self.params, x_star, prior_mean)
        prior_cov = self.cov(self.params, t_star)

        if len(y) == 0:
            return prior_mean, prior_cov

        t, rx = x
        obs_mean = self.mean(self.params, t)
        if treatment:
            obs_mean += self.tr(self.params, x, obs_mean)
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

            rx_logit = p[self.cf_param_key]
            p_rx = 1.0 / (1.0 + np.exp(-rx_logit)) # [0.5, 1)
            ln_p_rx = np.log(np.array([p_rx, 1 - p_rx]))
            
            for y, x in samples:
                # Outcome model
                fs = [_ln_p_rx + log_likelihood(p, y, x, mean_fn=self.mean, cov_fn=self.cov, tr_fn=tr) 
                        for _ln_p_rx, tr in zip(ln_p_rx, [self.tr, None])]
                f -= logsumexp(np.array(fs))

                # Action model
                #_, rx = x
                #n_rx = np.sum(rx)
                #f -= np.dot(ln_p_rx, np.array([n_rx, len(rx)-n_rx]))

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
    if tr_fn:
        m += tr_fn(params, x, m)
    c = cov_fn(params, t)

    ln_p_y = vec_mvn_logpdf(y, m, c)

    return ln_p_y
