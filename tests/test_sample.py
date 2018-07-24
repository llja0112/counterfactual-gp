import pytest
import numpy as np
from counterfactualgp.mean import LinearModel


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
