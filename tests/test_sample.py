import pytest
import autograd.numpy as np
from counterfactualgp.mean import linear_mean
from counterfactualgp.cov import iid_cov, se_cov
from counterfactualgp.treatment import DummyTreatment, Treatment
from counterfactualgp.mpp import BinaryActionModel
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
    mp = m(mp, data['training2'], params_only=True)
    print(mp)

    coef_ = np.round(mp['linear_mean_coef'], 2).tolist()
    assert coef_[0] == 0.45
    assert coef_[1] == -0.63 # bias affected the treatment 

    yhat = m(mp, np.array([1,2,3]))
    assert np.round(yhat, 8).tolist() == [-0.17919259, 0.266663, 0.7125186]


@pytest.mark.skip(reason="")
def test_gp_with_dummy_treatment(data):
    m = linear_mean(1)
    tr = []
    tr.append( (1.0, DummyTreatment()) )
    gp = GP(m, se_cov(a=1.0, l=1.0), tr, ac_fn=None)

    gp.fit(data['training2'], init = False)
    print(gp.params)
    mean_coef_ = np.round(gp.params['linear_mean_coef'], 2).tolist()
    assert mean_coef_[0] == 0.45

    y, x = data['testing1'][0]
    yhat, cov_hat = gp.predict(x, y, x)   
    assert np.max(np.abs(yhat - y)) < 2.0 # simulated treatment effect


@pytest.mark.skip(reason="")
def test_gp_with_treatment(data):
    m = linear_mean(1)
    tr = []
    tr.append( (1.0, Treatment(4.0)) )
    gp = GP(m, se_cov(a=1.0, l=1.0), tr, ac_fn=None)

    gp.fit(data['training2'], init = False)
    print(gp.params)
    assert gp.params['treatment'] != 0.0
    mean_coef_ = np.round(gp.params['linear_mean_coef'], 2).tolist()
    assert mean_coef_[0] == 0.49

    y, x = data['testing1'][0]
    yhat, cov_hat = gp.predict(x, y, x)   
    assert np.max(np.abs(yhat - y)) < 2.0 # simulated treatment effect
 

#@pytest.mark.skip(reason="")
def test_gp_with_binary_action(data):
    m = linear_mean(1)
    tr = []
    tr.append( (0.0, DummyTreatment()) )
    tr.append( (1.0, Treatment(4.0)) )
    ac = BinaryActionModel()
    gp = GP(m, se_cov(a=1.0, l=1.0), tr, ac_fn=ac)

    gp.fit(data['training2'], init = False)
    print(gp.params)
    assert gp.params['treatment'] != 0.0
    mean_coef_ = np.round(gp.params['linear_mean_coef'], 2).tolist()
    assert mean_coef_[0] == 0.48

    y, x = data['testing1'][0]
    yhat, cov_hat = gp.predict(x, y, x)   
    assert np.max(np.abs(yhat - y)) < 2.0 # simulated treatment effect
