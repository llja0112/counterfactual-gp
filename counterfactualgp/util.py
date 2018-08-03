import autograd.numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree

from counterfactualgp.lmm import LinearMixedModel, learn_lmm


def cluster_trajectories(data, basis, n_clusters, method='complete'):
    data = [(y, np.ones(len(x))[:, None], basis.design(x)) for y,(x,_) in data]
    lmm = learn_lmm(data)
    
    beta, Sigma, noise = lmm.param_copy()
    coef = np.array([lmm.posterior(*x)[0] for x in data])
    link = linkage(coef, method)
    clusters = cut_tree(link, n_clusters).ravel()

    cluster_coef = np.ndarray((n_clusters, coef.shape[1]))
    for k in range(n_clusters):
        w = coef[clusters == k].mean(axis=0)
        cluster_coef[k] = w
    
    return lmm, cluster_coef


def make_predict_samples(samples, t_star=None, rx_star=None, truncated_time=None, copy_truncated_rx=False):
    '''
    :return: y, x, x_star
    '''
    
    def _concat_x(t1, rx1, t2, rx2):
        t = np.concatenate([t1, t2])
        rx = np.concatenate([rx1, rx2])
        idx = np.argsort(t)
        return t[idx], rx[idx]
            
    for y,x in samples:
        t, rx = x

        if truncated_time is not None:
            _y = y[t <= truncated_time]
            _t = t[t <= truncated_time]
            _rx = rx[t <= truncated_time]
        else:
            _y, _t, _rx = y, t, rx
        
        if t_star is None:
            _t_star = t[:]
            _rx_star = rx[:]
        else:
            _t_star, _rx_star = _concat_x(t, rx, t_star, rx_star)
            
        yield _y, (_t, _rx), (_t_star, _rx_star)
