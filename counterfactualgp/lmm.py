import logging
import numpy as np
import scipy.linalg as la

from scipy.stats import multivariate_normal as mvn


class LinearMixedModel:
    """Linear mixed model.

    This class stores the parameters of a linear mixed model. It has
    two methods: one to compute the log likelihood of a subject's
    data, and the other to compute the posterior distribution over the
    subject's random effects.

    """

    def __init__(self, p1, p2):
        self._coef = np.zeros(p1)
        self._ranef_cov = np.eye(p2)
        self._noise_var = 1.0

    def param_copy(self):
        """Return the model's parameters.

        Returns
        -------
        A 3-tuple; the fixed effects, random effects covariance, and
        noise variance.

        """
        beta = np.array(self._coef)
        Sigma = np.array(self._ranef_cov)
        v = float(self._noise_var)
        return beta, Sigma, v

    def log_likelihood(self, y, X, Z):
        """Marginal log-likelihood of a subject's data.

        Parameters
        ----------
        y : Response vector.
        X : Fixed effects design matrix.
        Z : Random effects design matrix.

        Returns
        -------
        The marginal log-likelihood (a scalar).

        """
        # Because ranef has zero mean, and coef has zero variance,
        #+thus the mean is the mean of FE only, and var is of RE only.
        m = np.dot(X, self._coef)
        # cov(y) = cov(dot(X, basis_coef))
        S = np.dot(Z, np.dot(self._ranef_cov, Z.T))
        S += self._noise_var * np.eye(len(y))
        return mvn_logpdf(y, m, S)

    def posterior(self, y, X, Z):
        """Posterior over random effects given the subject's data.

        Parameters
        ----------
        y : Response vector.
        X : Fixed effects design matrix.
        Z : Random effects design matrix.

        Returns
        -------
        A 2-tuple; the mean and covariance of the random effects.

        """
        # OLS for linear regression.
        # Bayesian linear regression.
        # P(b|X,y) \approx P(y,X|b)P(b)
        # https://www.cs.utah.edu/~fletcher/cs6957/lectures/BayesianLinearRegression.pdf

        resid = y - np.dot(X, self._coef)
        P = la.inv(self._ranef_cov) + np.dot(Z.T, Z) / self._noise_var
        S = la.inv(P)
        m = la.solve(P, np.dot(Z.T, resid) / self._noise_var)
        return m, S


def learn_lmm(dataset, maxiter=500, tol=1e-5):
    """Fit parameters of an LMM.

    Parameters
    ----------
    dataset : List of 3-tuples (see Notes for details).
    maxiter : Maximum number of EM iterations.
    tol : Minimum relative improvement required to stop early.

    Returns
    -------
    A LinearMixedModel object.

    Notes
    -----
    Each tuple in the dataset should contain an array of responses,
    the fixed effects design matrix, and the random effects design
    matrix for a subject.

    """
    objective = lambda lmm: np.mean([lmm.log_likelihood(*d) for d in dataset])

    p1 = dataset[0][1].shape[1]
    p2 = dataset[0][2].shape[1]
    lmm = LinearMixedModel(p1, p2)

    ss1 = 0.0
    ss2 = 0.0

    for y, X, _ in dataset:
        ss1 += np.dot(X.T, X)
        ss2 += np.dot(X.T, y)

    lmm._coef[:] = np.linalg.solve(ss1, ss2)

    logl = objective(lmm)

    for iteration in range(maxiter):
        lmm._coef, lmm._ranef_cov, lmm._noise_var = em_step(dataset, lmm)

        logl_old = logl
        logl = objective(lmm)

        delta = (logl - logl_old) / np.abs(logl_old)

        msg = 'lmm: iteration={:05d} LL={:13.8f}, dLL={:10.8f}'
        logging.debug(msg.format(iteration, logl, delta))

        if delta < tol:
            break

    return lmm


def em_step(dataset, lmm):
    """Complete one expectation and one maximization step.

    Parameters
    ----------
    dataset : List of tuples; observations and design matrices.
    lmm : A linear mixed model object.

    Returns
    -------
    A 3-tuple; the fixed effects, random effects covariance, and
    noise variance.

    """
    ss1 = 0.0
    ss2 = 0.0
    ss3 = 0.0
    ss4 = 0.0
    ss5 = 0.0

    for y, X, Z in dataset:
        m_i, S_i = lmm.posterior(y, X, Z)

        ss1 += np.dot(X.T, X)
        ss2 += np.dot(X.T, y - np.dot(Z, m_i))

        ss3 += S_i + np.outer(m_i, m_i)

        ss4 += len(y)

    b = la.solve(ss1, ss2)
    S = ss3 / len(dataset)

    for y, X, Z in dataset:
        m_i, S_i = lmm.posterior(y, X, Z)
        ss5 += np.sum((y - np.dot(X, b) - np.dot(Z, m_i))**2)
        ss5 += np.diag(np.dot(Z.T, Z) * S_i).sum()

    v = ss5 / ss4

    return b, S, v


def mvn_logpdf(x, m, c):
    q = -0.5 * np.dot(x - m, np.linalg.solve(c, x - m))
    z = 0.5 * np.linalg.slogdet(2 * np.pi * c)[1]
    return q - z
