import autograd.numpy as np


def action_log_likelihood(rx, ln_p_a, continuous=False):
    '''
    P(a, z_a|t)
    z_a \in {0,1}
    a \in {1,...,k} or R
    In the case of continuous valued treatment,
    we assume P(a|t,z_a=1) = const, thus inproper density.
    '''
    if ln_p_a.shape[0] == 1:
        return 0

    if continuous:
        na = np.sum(rx > 0.0)
        n_rx = [len(rx)-na, na]
    else:
        n_rx = [np.sum(rx == i) for i in range(ln_p_a.shape[0])]
    return np.dot(ln_p_a.T, np.array(n_rx))


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
