from data_perturb import CDataPerturb
import numpy as np

class CDataPerturbRandom(CDataPerturb):
    MIN_VALUE = 0
    MAX_VALUE = 255

    def __init__(self, min_value=MIN_VALUE, max_value=MAX_VALUE, k=100):
        self._min_value = min_value
        self._max_value = max_value
        self._k = k

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

    def data_perturbation(self, x):
        x_perturbed = np.zeros(shape=x.shape)
        noise = np.random.uniform(self._min_value,self._max_value,self._k)
        x_perturbed = (x[: self._k] + noise) % self._max_value

        return x_perturbed
