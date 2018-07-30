import pytest
import autograd.numpy as np
from counterfactualgp.mean import Linear, LinearWithBsplinesBasis
from counterfactualgp.cov import iid_cov, se_cov
from counterfactualgp.treatment import DummyTreatment, Treatment
from counterfactualgp.mpp import BinaryActionModel
from counterfactualgp.gp import GP


@pytest.fixture
def linear_data():
    import pickle
    with open('dataset/data_set_linear.pkl', 'rb') as fin:
        return pickle.load(fin)


@pytest.fixture
def data():
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
