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

    @property
    def k(self):
        return self._K

    @k.setter
    def k(self, value):
        self._K = value

    def data_perturbation(self, x, normalized = False):
        idx = np.array(list(range(0, x.size)))
        np.random.shuffle(idx)
        idx = idx[:self._k]
        x_perturbed = x.copy()

        divider = 1 if not normalized else 255
        
        random_pixels = np.random.randint(
            low=self._min_value,
            high=self._max_value + 1, size=self._k)/divider
        
        x_perturbed[idx] = random_pixels
        return x_perturbed
