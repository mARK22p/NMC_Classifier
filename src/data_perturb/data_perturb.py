from abc import ABC, abstractmethod

import numpy as np


class CDataPerturb(ABC):

    def __init__(self, X):
        self._X = X

    @abstractmethod
    def data_perturbation(self, x):
        raise NotImplementedError("You must implement data_pertubation in child classes")

    def perturb_dataset(self):
        X_perturbed = np.zeros(shape=self._X.shape)

        for i in range(self._X.shape[0]):
            X_perturbed[i, :] = self.data_perturbation(self._X[i])

        return X_perturbed
            #data_perturbation(_X[i])