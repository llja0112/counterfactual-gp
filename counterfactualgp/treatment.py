import autograd.numpy as np


def treatment(effects_window):

    def treatment_params():
        return {
            'treatment': np.zeros(1),
            'effects_window_F': np.array([effects_window]),
        }

    def treat(params, x, y=None, stack=True):
        '''
        This treatment function is only added on to the mean value.
        Its value may depend on the mean value.

        :param params:
        :param x: (t, rx)
        :param y: mean value
        '''
        c = params['treatment']
        w = params['effects_window_F']
    
        t, rx = x
        t_rx = t[rx == 1]
        d = t[:, None] - t_rx[None, :] # shape (t, t_rx)
        treated = (d > 0) & (d <= w) # w for effects_window

        if stack:
            return c * np.sum(treated, axis=1) # shape (t,)
        else:
            return c * np.any(treated, axis=1).astype(float)

    def func(*args, **kwargs):
        params = args[0] if len(args) > 0 else None
        samples = args[1] if len(args) > 0 else None

        if kwargs.get('params_only', None):
            return treatment_params()
        else:
            return treat(*args, **kwargs)

    return func
