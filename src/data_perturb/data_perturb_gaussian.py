from data_perturb import CDataPerturb
import numpy as np

class CDataPerturbGaussian(CDataPerturb):
    def __init__(self):
        pass

    def data_perturbation(self, x):
        noise = np.random.normal(0, 2, len(x))  # μ = 0, σ = 2, size = length of x or y. Choose μ and σ wisely.

