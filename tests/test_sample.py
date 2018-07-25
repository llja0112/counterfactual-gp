import pytest
import autograd.numpy as np
from counterfactualgp import mean
from counterfactualgp.gp import GP, log_likelihood


@pytest.fixture
def data():
    import pickle
    with open('dataset/data_set_linear.pkl', 'rb') as fin:
        return pickle.load(fin)


def test_pass():
    assert True, "dummy sample test"


def test_mean_linear(data):
    m = mean.linear_params(1)
    m = mean.linear_fit_params(m, data['training'])

    # [0.10835268957176855, 0.9366460911796043]
    coef_ = np.round(m['linear_mean_coef'], 1).tolist()
    assert coef_[0] == 0.1
    assert coef_[1] != 0.5 # bias changes after treatment 

    yhat = mean.linear_predict(m, np.array([1,2,3]))
    assert np.round(yhat, 8).tolist() == [1.04499878, 1.15335147, 1.26170416]
 

def test_gp_train(data):
    gp = GP(1)
    gp.fit(data['training'], init = False)
    print(gp.params)
    mean_coef_ = np.round(gp.params['linear_mean_coef'], 1).tolist()
    assert mean_coef_[0] == 0.1
    assert mean_coef_[1] != 0.5 # bias changes after treatment 
