import numpy as np


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


def make_missed_obs_samples(samples):
    new_samples = []
    for y, (t, rx) in samples:
        idx = rx == 1
        y_new = np.array(y)
        y_new[idx] = y[idx] + np.random.choice([np.nan, 0], len(y[idx]), p=[0.3, 0.7])
        new_samples.append((y_new, (t, rx)))

    return new_samples
