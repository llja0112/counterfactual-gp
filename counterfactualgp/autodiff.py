import autograd.numpy as np

from autograd.scipy.stats import multivariate_normal as mvn
from autograd.scipy.misc import logsumexp


def packing_funcs(params):
    keys = sorted(params.keys())
    sizes = {k:params[k].size for k in keys}
    shapes = {k:params[k].shape for k in keys}

    def pack(params):
        flat = [params[k].ravel() for k in keys]
        return np.concatenate(flat)

    def unpack(packed):
        params = {}
        num_read = 0

        for k in keys:
            n = sizes[k]
            s = shapes[k]
            params[k] = packed[num_read:(num_read + n)].reshape(s)
            num_read += n

        return params

    return pack, unpack


def vec_mvn_logpdf(x, m_vec, cov):
    '''Multiple classes version of log PDF of Multivariate normal distribution'''

    m_vec = np.atleast_2d(m_vec)
    logdet = np.linalg.slogdet(cov)[1]
    d = m_vec.shape[1]
    r = x - m_vec
    q = np.sum(np.linalg.solve(cov, r.T).T * r, axis=1) # shape (n_class,)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + q)
