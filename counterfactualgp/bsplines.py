import autograd.numpy as np

from scipy.interpolate import splev


class BSplines:
    def __init__(self, low, high, num_bases, degree,
                 x=None, boundaries='stack'):
        '''
        B spline based at `num_bases` dimension unit vectors.
        Given a point within the knots range, B spline moves from
        (1,0,...,0) to (0,0,...,1).
        '''

        self._low = low
        self._high = high
        self._num_bases = num_bases
        self._degree = degree

        use_quantiles_as_knots = x is not None

        if use_quantiles_as_knots:
            knots = _quantile_knots(low, high, x, num_bases, degree)
        else:
            knots = _uniform_knots(low, high, num_bases, degree)

        if boundaries == 'stack':
            self._knots = _stack_pad(knots, degree)
        elif boundaries == 'space':
            self._knots = _space_pad(knots, degree)

        # (knots, coefficients, degree of the spline)
        self._tck = (self._knots, np.eye(num_bases), degree)

    @property
    def dimension(self):
        return self._num_bases

    def design(self, x):
        return np.array(splev(np.atleast_1d(x), self._tck)).T # shape (num_bases, len(x))


def _uniform_knots(low, high, num_bases, degree):
    num_interior_knots = num_bases - (degree + 1)
    knots = np.linspace(low, high, num_interior_knots + 2)
    return np.asarray(knots)


def _quantile_knots(low, high, x, num_bases, degree):
    num_interior_knots = num_bases - (degree + 1)
    clipped = x[(x >= low) & (x <= high)]
    knots = np.percentile(clipped, np.linspace(0, 100, num_interior_knots + 2))
    knots = [low] + list(knots[1:-1]) + [high]
    return np.asarray(knots)


def _stack_pad(knots, degree):
    knots = list(knots)
    knots = ([knots[0]] * degree) + knots + ([knots[-1]] * degree)
    return knots


def _space_pad(knots, degree):
    knots = list(knots)
    d1 = knots[1] - knots[0]
    b1 = np.linspace(knots[0] - d1 * degree, knots[0], degree + 1)
    d2 = knots[-1] - knots[-2]
    b2 = np.linspace(knots[-1], knots[-1] + d2 * degree, degree + 1)
    return list(b1) + knots[1:-1] + list(b2)
