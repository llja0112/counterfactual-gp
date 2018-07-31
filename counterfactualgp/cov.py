import autograd.numpy as np


def iid_cov(*args, **kwargs):
    def get_params():
        return {
            'ln_cov_y': np.zeros(1),
        }

    def iid(params, t1, t2=None, eps=1e-3):
        v_y = np.exp(params['ln_cov_y'])
        symmetric = t2 is None

        if symmetric:
            cov = (v_y + eps) * np.eye(len(t1))
        else:
            cov = np.zeros((len(t1), len(t2)))

        return cov

    if kwargs.get('params_only', None):
        return get_params()
    else:
        return iid(*args, **kwargs)


def linear_cov(basis=None, eps=1e-3):
    def get_params():
        return {
            'ln_cov_y': np.zeros(1),
            'ln_cov_w': np.zeros(1),
        }

    def linear_cov(basis, params, t1, t2=None, eps=1e-3):
        v_w = np.exp(params['ln_cov_w'])
        v_y = np.exp(params['ln_cov_y'])

        if t2 is None:
            t2 = t1
            symmetric = True
        else:
            symmetric = False

        if basis:
            b1 = basis.design(t1)
            b2 = basis.design(t2)
            cov = v_w * np.dot(b1, b2.T)
        else:
            cov = v_w * np.dot(t1, t2.T)

        if symmetric:
            cov += (v_y + eps) * np.eye(len(t1))

        return cov

    def func(*args, **kwargs):
        if kwargs.get('params_only', None):
            return get_params()
        else:
            return linear_cov(basis, *args, **kwargs)

    return func


def se_cov(a, l):
    '''Squared Exponential Kernel'''

    def get_params(ln_a, ln_l):
        return {
            'ln_cov_a_F': np.array([ln_a]),
            'ln_cov_l_F': np.array([ln_l]),
            'ln_cov_y': np.zeros(1),
            }

    def se(params, t1, t2=None, eps=1e-3):
        v_y = np.exp(params['ln_cov_y'])
        a = np.exp(params['ln_cov_a_F'])
        l = np.exp(params['ln_cov_l_F'])

        symmetric = t2 is None
        if symmetric:
            t2 = t1

        D = np.expand_dims(t1, 1) - np.expand_dims(t2, 0)
        cov = a**2 * np.exp(-0.5 * D**2 / l**2)
        if symmetric:
            cov += (v_y + eps) * np.eye(len(t1))

        return cov

    def func(*args, **kwargs):
        if kwargs.get('params_only', None):
            return get_params(ln_a, ln_l)
        else:
            return se(*args, **kwargs)

    ln_a = np.log(a)
    ln_l = np.log(l)

    return func
