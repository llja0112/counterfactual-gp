import pytest
import autograd.numpy as np
from counterfactualgp.mean import LinearModel
from counterfactualgp.gp import GP, log_likelihood


@pytest.fixture
def data():
    import pickle
    with open('dataset/data_set_linear.pkl', 'rb') as fin:
        return pickle.load(fin)


def test_pass():
    assert True, "dummy sample test"


def test_linearmodel(data):
    m = LinearModel(1)
    m.fit(data['training'])
    coef_ = np.round(m.coef, 1).tolist()
    assert coef_[0] == 0.1
    assert coef_[1] != 0.5 # bias changes after treatment 
    

def test_likelihood(data):
    y, x = data['training'][0]
    p = {'mean_coef': np.array([0.10838247, 0.93616695]), 'ln_cov_y': np.array([-1.2293894])}
    f = log_likelihood(p, 1, y, x)
    assert round(f[0], 1) == round(-10.18989142, 1)


def test_gp_train(data):
    gp = GP(1)
    gp.fit(data['training'], init = False)
    print(gp.params)
    # {'mean_coef': array([0.10838247, 0.93616695]), 'ln_cov_y': array([-1.2293894])}
    mean_coef_ = np.round(gp.params['mean_coef'], 1).tolist()
    assert mean_coef_[0] == 0.1
    assert mean_coef_[1] != 0.5 # bias changes after treatment 
