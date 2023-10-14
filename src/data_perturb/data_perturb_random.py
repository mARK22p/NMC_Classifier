from data_perturb import CDataPerturb
import numpy as np

class CDataPerturbRandom(CDataPerturb):
    def __init__(self, min_value=0, max_value=255, k=100):
        self._min_value = min_value
        self._max_value = max_value
        self._k = k

    def data_perturbation(self, x):
        print('CDataPerturbationRandom data_perturbation')

        perturbIndexes = np.random.uniform(self._min_value,self._max_value,self._k)

        pass
