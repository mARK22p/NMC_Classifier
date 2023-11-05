from data_perturb import CDataPerturb
import numpy as np

class CDataPerturbGaussian(CDataPerturb):
    MIN_VALUE = 0
    MAX_VALUE = 255
    def __init__(self, min_value = MIN_VALUE, max_value = MAX_VALUE, sigma=100):
        self._min_value = min_value
        self._max_value = max_value
        self._sigma = sigma

    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self,value):
        if value < 0 or value > self.MAX_VALUE :
            raise ValueError(f'min_value parameter should between {0} and {self.MAX_VALUE}')
        self._min_value = value

    @property
    def max_value(self):
        return self._max_value

    @max_value.setter
    def max_value(self, value):
        if value < 0 or value > self.MAX_VALUE:
            raise ValueError(f'max_value parameter should between {0} and {self.MAX_VALUE}')
        self._max_value = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    def data_perturbation(self, x):
        noise =  np.round(self.sigma * np.random.randn(x.ravel().size)) #
        xp = x + noise
        xp[xp < self.min_value] = self.min_value  # projections on box [0, 255]
        xp[xp > self.max_value] = self.max_value
        return xp

