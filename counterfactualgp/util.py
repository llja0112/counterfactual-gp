import autograd.numpy as np


def decompose_rank1_mat(mat):
    r = mat[0,:]
    c = mat[:,0]
    return (c / np.sum(c)), (r / np.sum(r))
