import pytest
import autograd.numpy as np
from scipy.stats import linregress

from counterfactualgp.mean import Linear, LinearWithBsplinesBasis
from counterfactualgp.cov import iid_cov, se_cov, linear_cov
from counterfactualgp.treatment import DummyTreatment, Treatment
from counterfactualgp.mpp import BinaryActionModel
from counterfactualgp.bsplines import BSplines
from counterfactualgp.gp import GP
from counterfactualgp.util import make_predict_samples, cluster_trajectories


@pytest.fixture
def linear_data():
    import pickle
    with open('dataset/data_set_linear.pkl', 'rb') as fin:
        return pickle.load(fin)


@pytest.fixture
def bspline_data():
    import pickle
    with open('dataset/data_set_bspline_3classes.pkl', 'rb') as fin:
        return pickle.load(fin)


def test_pass():
    assert True, "dummy sample test"


def test_mean_linear(linear_data):
    m = Linear(1)
    mp = m(params_only=True)
    mp = m(mp, linear_data['training2'], params_only=True)
    print(mp)

    coef_ = np.round(mp['linear_mean_coef'], 2).tolist()
    assert coef_[0] == 0.45
    assert coef_[1] == -0.63 # bias affected the treatment 

    yhat = m(mp, np.array([1,2,3]))
    assert np.round(yhat, 8).tolist() == [-0.17919259, 0.266663, 0.7125186]


def test_lmm(bspline_data):
    low, high = bspline_data['xlim']
    num_bases = 5
    bsplines_degree = 3
    basis = BSplines(low, high, num_bases, bsplines_degree, boundaries='space')

    n_clusters = len(bspline_data['class_prob'])

    lmm, cluster_coef = cluster_trajectories(bspline_data['training2'], basis, n_clusters, method='complete')
    beta, Sigma, noise = lmm.param_copy()

    t = np.linspace(low, high, num=100)
    slopes = []
    for k in range(n_clusters):
        yhat = np.dot(np.ones(len(t))[:, None], beta) + np.dot(basis.design(t), cluster_coef[k])
        slope, intercept, r_value, p_value, std_err = linregress(t, yhat)
        slopes.append(slope)

    assert np.round(slopes, 4).tolist() == [-0.042, 0.0043, -0.0827]


#@pytest.mark.skip(reason="")
def test_gp(bspline_data):
    low, high = bspline_data['xlim']
    num_bases = 5
    bsplines_degree = 3
    basis = BSplines(low, high, num_bases, bsplines_degree, boundaries='space')

    n_clusters = len(bspline_data['class_prob'])
    random_basis = np.random.multivariate_normal(np.zeros(num_bases), 0.1*np.eye(num_bases), n_clusters)

    n_train = bspline_data['n_train']
    truncated_time = bspline_data['truncated_time']

    m = []
    for i in range(n_clusters):
        m.append(LinearWithBsplinesBasis(basis, no=i, init=random_basis[i]))
    tr = []
    tr.append( (0.0, DummyTreatment()) )
    tr.append( (1.0, Treatment(2.0)) )
    ac = BinaryActionModel()
    mcgp = GP(m, linear_cov(basis), tr, ac_fn=ac)
    mcgp.fit(bspline_data['training2'], options={'maxiter':1})
    print(mcgp.params)

    _test_gp_params(mcgp.params)
    _test_gp_prediction(mcgp, bspline_data['testing1'][0:20], truncated_time)
    _test_gp_prediction(mcgp, bspline_data['testing1'][0:20], truncated_time, [0])


def _test_gp_params(p):
    assert p['action'].tolist() != [0.0]
    assert p['treatment'].tolist() != [0.0]
    assert p['treatment'].tolist() != [0.0]
    assert np.all(p['classes_prob_logit'] > -5) # logP < -5, then P<0.01


def _test_gp_prediction(m, data, truncated_time, exclude_ac=[]):
    _samples = make_predict_samples(data, None, None, truncated_time)
    s = 0.0
    for (y, x), (_y, _x, _x_star) in zip(data, _samples):
        yhat, cov_hat = m.predict(_x_star, _y, _x, exclude_ac)
        _t_star, _rx_star = _x_star
        idx = _t_star > truncated_time
        s += np.sum((yhat - y)[idx] **2) / len(y[idx])

        p_a, p_mix = m.class_posterior(_y, _x, exclude_ac)
        assert len(p_a) == 2 - len(exclude_ac)
        assert np.round(np.sum(p_a), 0) == 1.0
        assert np.round(np.sum(p_mix), 0) == 1.0
    
    mse = s / len(data)
    print(mse)
    assert True
