import numpy as np


class Trajectory:
    def __init__(self, func):
        self.func = func

    def value(self, x):
        return self.func(x)


class ObservationTimes:
    def __init__(self, low, high, avg_n_obs):
        self.low = low
        self.high = high
        self.avg_n_obs = avg_n_obs

    def sample(self, rng):
        n_obs = 1 + rng.poisson(self.avg_n_obs - 1)
        return np.sort(rng.uniform(self.low, self.high, n_obs))


def _ou_cov(t1, t2, ln_a, ln_l):
    '''A SE alike kernel'''

    a = np.exp(ln_a)
    l = np.exp(ln_l)
    D = np.expand_dims(t1, 1) - np.expand_dims(t2, 0)
    return a * np.exp(-np.abs(D) / l)


def sample_trajectory(traj, obs_proc, ln_a, ln_l, noise_scale, rng):
    t = obs_proc.sample(rng)
    C = _ou_cov(t, t, ln_a, ln_l) + noise_scale**2 * np.eye(len(t))
    y = traj.value(t) + rng.multivariate_normal(np.zeros(len(t)), C)
    # y = traj.value(t) + noise_scale * rng.normal(size=len(t))
    return (y, t)


class TreatmentPolicy:
    def __init__(self, history_window, weight, bias, effect_window, effect):
        self.history_window = history_window
        self.weight = weight
        self.bias = bias
        self.effect_window = effect_window
        self.effect = effect

    def sample_treatment(self, y, t, rng):
        t0 = t[-1]
        time_to = t0 - t
        in_window = time_to <= self.history_window # average last y's
        avg = np.mean(y[in_window])
        prob_rx = sigmoid(self.weight * avg + self.bias)
        return rng.binomial(1, prob_rx)

    def treat(self, y, t, treated, t0, stack):
        y_rx = np.array(y)
        t_rx = np.array(t)

        in_future = t > t0
        in_range = t <= (t0 + self.effect_window)
        treat = (in_future & in_range)

        if stack:
            y_rx += self.effect * treat
        else:
            y_rx += self.effect * (treat & (~treated))

        return y_rx, t_rx, (treated | treat)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def treat_data_set(samples, policy, rng):
    return [treat_sample(y, t, policy, rng) for y, t in samples]


def treat_sample(y, t, policy, rng, stack=False):
    y_rx = np.array(y)
    t_rx = np.array(t)
    treated = np.zeros(len(t), dtype=bool)
    rx = np.zeros(len(t))

    for i, t0 in enumerate(t):
        rx[i] = policy.sample_treatment(y_rx[:(i+1)], t_rx[:(i+1)], rng)

        if rx[i] == 1:
            y_rx, t_rx, treated = policy.treat(y_rx, t_rx, treated, t0, stack)

    return y_rx, (t_rx, rx)
