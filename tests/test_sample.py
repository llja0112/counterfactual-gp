import pytest
import autograd.numpy as np
from counterfactualgp.mean import linear_mean
from counterfactualgp.cov import iid_cov, se_cov
from counterfactualgp.gp import GP


@pytest.fixture
def data():
    import pickle
    with open('dataset/data_set_linear.pkl', 'rb') as fin:
        return pickle.load(fin)


def test_pass():
    assert True, "dummy sample test"


def test_mean_linear(data):
    m = linear_mean(1)
    mp = m(params_only=True)
    print(mp)
    mp = m(mp, data['training'], params_only=True)

    # [0.10835268957176855, 0.9366460911796043]
    coef_ = np.round(mp['linear_mean_coef'], 1).tolist()
    assert coef_[0] == 0.1
    assert coef_[1] != 0.5 # bias changes after treatment 

    yhat = m(mp, np.array([1,2,3]))
    assert np.round(yhat, 8).tolist() == [1.04499878, 1.15335147, 1.26170416]
 

def test_gp(data):
    m = linear_mean(1)
    #gp = GP(m, iid_cov)
    gp = GP(m, se_cov(a=1.0, l=1.0))
    gp.fit(data['training'], init = False)
    print(gp.params)
    mean_coef_ = np.round(gp.params['linear_mean_coef'], 1).tolist()
    assert mean_coef_[0] == 0.1
    assert mean_coef_[1] != 0.5 # bias changes after treatment 

    y, x = data['testing'][0]
    t, rx = x
    yhat, cov_hat = gp.predict((t, rx), y, x)   
    # assert np.round(np.sum(yhat - y), 2) == 0.64
    assert np.round(np.sum(yhat - y), 2) == 0.01
