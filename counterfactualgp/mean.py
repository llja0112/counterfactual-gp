'''
Models have to be written in separate functions instead of classes,
for the sake of autograd uasge.
'''


import autograd.numpy as np


def linear_mean(degree):
    def linear_params(degree):
        return {
            'linear_mean_coef': np.zeros(degree+1),
        }

    def linear_fit_params(params, samples):
        '''Pooled least square fit'''

        degree = params['linear_mean_coef'].shape[0] - 1
        t = np.concatenate([t for y, (t, rx) in samples])
        y = np.concatenate([y for y, (t, rx) in samples])
        params['linear_mean_coef'] = np.polyfit(t, y, degree)
        return params

    def linear_predict(params, x):
        '''
        Self-made, since np.poly1d can't be optimized.
        '''
        # return np.poly1d(params['linear_mean_coef'])(t)
        sum = 0.0
        degree = params['linear_mean_coef'].shape[0] - 1
        for d in range(degree+1):
            sum += params['linear_mean_coef'][d] * np.power(x, degree-d) 
        return sum

    def func(*args, **kwargs):
        '''
        :param params:
        :param samples:
        :param kwargs:
        '''
        params = args[0] if len(args) > 0 else None
        samples = args[1] if len(args) > 0 else None

        if kwargs.get('params_only', None):
            if samples:
                return linear_fit_params(params, samples)
            else:
                return linear_params(degree)
        else:
            return linear_predict(*args, **kwargs)

    return func
