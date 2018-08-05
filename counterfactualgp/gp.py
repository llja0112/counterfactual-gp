import autograd
import autograd.numpy as np
import sys
import pickle

from autograd.scipy.misc import logsumexp
from numpy.linalg.linalg import LinAlgError
from scipy.optimize import minimize

from counterfactualgp.autodiff import packing_funcs, vec_mvn_logpdf


class GP:
    def __init__(self, mean_fn=[], cov_fn=None, tr_fns=[], ac_fn=None):
        self.params = {}

        self.mean = mean_fn
        self.n_classes = len(self.mean)
        for m in self.mean:
            self.params.update(m(params_only=True))
        if self.n_classes == 1:
            self.mixture_param_key = 'classes_prob_logit_F'
            self.params[self.mixture_param_key] = np.array([1.0])
        else: # Mixture of GPs
            self.mixture_param_key = 'classes_prob_logit'
            self.params[self.mixture_param_key] = np.zeros(self.n_classes)

        self.cov = cov_fn
        if self.cov: self.params.update(self.cov(params_only=True))

        if tr_fns:
            for _, tr in tr_fns:
                self.params.update(tr(params_only=True))
        else:
            # Dummy treatment
            tr_fns = [(1.0, lambda *args, **kwargs: 0)]
        self.tr = [tr for _,tr in tr_fns]
        self.action = lambda *args, **kwargs: [prob for prob,_ in tr_fns]

        # Use action model to replace fixed probs
        if ac_fn:
            self.action = ac_fn
            self.params.update(self.action(params_only=True))

    def predict(self, x_star, y, x, exclude_ac=[]):
        l = len(x_star[0])
        c = np.zeros([l, l])

        # exclude actions
        include_idx = ~np.in1d(range(len(self.tr)), exclude_ac)
        tr_fns = [tr for b, tr in zip(include_idx, self.tr) if b]
        ln_p_am = self._class_posterior(y, x, exclude_ac).ravel()

        ms = []
        for mn in self.mean:
            for tr in tr_fns:
                _m, _c = self._predict(x_star, y, x, mean_fn=mn, treatment=tr)
                ms.append(_m)
                c = _c # all covariance matrix are the same

        ms = [p*_m for p,_m in zip(np.exp(ln_p_am), ms)]
        return np.sum(ms, axis=0), c

    def _class_posterior(self, y, x, exclude_ac):
        ln_p_a, ln_p_mix = self.class_prior()

        include_idx = ~np.in1d(range(len(self.tr)), exclude_ac)
        ln_p_a = ln_p_a[include_idx]
        tr_fns = [tr for b, tr in zip(include_idx, self.tr) if b]

        mixture =  log_likelihood(self.params, y, x, self.mean, self.cov, tr_fns, ln_p_a, ln_p_mix)
        return mixture - logsumexp(mixture)

    def class_posterior(self, y, x, exclude_ac=[]):
        '''
        Note: self._class_posterior is not a rank-1 matrix
        :return: p_a, p_mix
        '''
        include_idx = ~np.in1d(range(len(self.tr)), exclude_ac)
        ln_p_am = self._class_posterior(y, x, exclude_ac)

        mat = np.exp(ln_p_am.reshape(self.n_classes, -1))
        return np.sum(mat, axis=0), np.sum(mat, axis=1)

    def class_prior(self):
        ln_p_a = np.log(self.action(self.params)) # individual- and time-invariant
        logits_mix = self.params[self.mixture_param_key]
        ln_p_mix = logits_mix - logsumexp(logits_mix)

        return ln_p_a, ln_p_mix

    def _predict(self, x_star, y, x, mean_fn, treatment):
        t_star, rx_star = x_star
        prior_mean = mean_fn(self.params, t_star)
        prior_mean += treatment(self.params, x_star, prior_mean)
        prior_cov = self.cov(self.params, t_star)

        if len(y) == 0:
            return prior_mean, prior_cov

        t, rx = x
        y_idx = ~np.isnan(y)
        obs_mean = mean_fn(self.params, t)
        obs_mean += treatment(self.params, x, obs_mean)
        obs_cov = self.cov(self.params, t[y_idx])

        cross_cov = self.cov(self.params, t_star, t[y_idx])

        alpha = np.linalg.solve(obs_cov, cross_cov.T).T
        mean = prior_mean + np.dot(alpha, y[y_idx] - obs_mean[y_idx])
        cov = prior_cov - np.dot(alpha, cross_cov.T)

        return mean, cov

    def fit(self, samples, options={}):
        trainable_params = dict([(k,v) for k,v in self.params.items() if not k.endswith('_F')])
        fixed_params = dict([(k,v) for k,v in self.params.items() if k.endswith('_F')])
        pack, unpack = packing_funcs(trainable_params)

        def obj(w):
            p = unpack(w)
            p.update(fixed_params)
            f = 0.0

            # ln_p_a, ln_p_mix = self.class_prior()
            ln_p_a = np.log(self.action(p)) # individual- and time-invariant
            logits_mix = p[self.mixture_param_key]
            ln_p_mix = logits_mix - logsumexp(logits_mix)

            for y, x in samples:
                # Outcome model
                mixture =  log_likelihood(p, y, x, self.mean, self.cov, self.tr, ln_p_a, ln_p_mix)
                f -= logsumexp(np.array(mixture))

                # Action model
                # TODO: continuous action models
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

        solution = minimize(obj, pack(self.params), jac=grad, method='BFGS', callback=callback, options=options)
        self.params = unpack(solution['x'])
        self.params.update(fixed_params)

    def dump_model(self, f):
        m = {
            'params'   : self.params,
            'mean_fn'  : self.mean,
            'cov_fn'   : self.cov,
            'tr_fn'    : self.tr,
            'ac_fn'    : self.action,
        }
        with open(f, 'wb') as fout:
            pickle.dump(m, fout)

    def load_model(self, f):
        with open(f, 'rb') as fin:
            m = pickle.load(fin)
            self.params = m['params']
            self.mean = m['mean_fn']
            self.cov = m['cov_fn']
            self.tr = m['tr_fn']
            self.action = m['ac_fn']


def log_likelihood(params, y, x, mean_fns, cov_fn, tr_fns, ln_p_a, ln_p_mix):
    mixture = []
    for m, _ln_p_mix in zip(mean_fns, ln_p_mix):
        for tr, _ln_p_a in zip(tr_fns, ln_p_a):
            mixture.append(_ln_p_a + _ln_p_mix + _log_likelihood(params, y, x, m, cov_fn, tr))

    return mixture


def _log_likelihood(params, y, x, mean_fn, cov_fn, tr_fn):
    t, rx = x
    y_idx = ~np.isnan(y)
    m = mean_fn(params, t)
    m += tr_fn(params, x, m)
    c = cov_fn(params, t[y_idx])

    ln_p_y = vec_mvn_logpdf(y[y_idx], m[y_idx], c)

    return ln_p_y
