import autograd.numpy as np


def DummyTreatment():

    def get_params():
        return {}

    def treat(params, x, m, stack=True):
        return 0

    def func(*args, **kwargs):
        if kwargs.get('params_only', None):
            return get_params()
        else:
            return treat(*args, **kwargs)

    return func


def Treatment(effects_window):

    def get_params():
        return {
            'treatment': np.zeros(1),
            'effects_window_F': np.array([effects_window]),
        }

    def treat(params, x, m, stack=True):
        '''
        This treatment function is only added on to the mean value.
        Its value may depend on the mean value.

        :param params:
        :param x: (t, rx)
        :param m: mean value
        '''
        c = params['treatment']
        w = params['effects_window_F']
    
        t, rx = x
        t_rx = t[rx == 1]
        d = t[:, None] - t_rx[None, :] # (t, t_rx)
        treated = (d > 0) & (d <= w) # w for effects_window

        if stack:
            return c * np.sum(treated, axis=1) # (t,)
        else:
            return c * np.any(treated, axis=1).astype(float)

    def func(*args, **kwargs):
        if kwargs.get('params_only', None):
            return get_params()
        else:
            return treat(*args, **kwargs)

    return func
