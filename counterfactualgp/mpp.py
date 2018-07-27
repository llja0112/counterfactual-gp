import autograd.numpy as np


def BinaryActionModel():
    def get_params():
        return {
            'action': np.zeros(1),
        }

    def get_action_prob(params):
        '''
        :return: [P(a=0), P(a=1), ... P(a=k)]
        '''
        logit = params['action']
        p_a = 1.0 / (1.0 + np.exp(-logit)) # [0.5, 1)
        return np.array([1 - p_a, p_a])

    def func(*args, **kwargs):
        '''
        :param params:
        '''
        params = args[0] if len(args) > 0 else None

        if kwargs.get('params_only', None):
            return get_params()
        else:
            return get_action_prob(params)

    return func
