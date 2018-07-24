import autograd.numpy as np


class LinearModel(object):
    def __init__(self, degree):
        self.degree = degree
        self._coef = np.zeros(degree+1)

    @property
    def coef(self):
        return self._coef

    def predict(self, x):
        return np.poly1d(x)

    def fit(self, samples):
        '''Pooled least square fit'''

        t = np.concatenate([t for y, (t, rx) in samples])
        y = np.concatenate([y for y, (t, rx) in samples])
        self._coef = np.polyfit(t, y, self.degree)
