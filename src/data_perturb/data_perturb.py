from abc import ABC, abstractmethod

import numpy as np


class CDataPerturb(ABC):
    """
    By adding noise to your synthetic data, you give your model the opportunity to generalise on the noisy data
    and learn the underlying patterns in the data instead of fitting each and every data-point. This could
    prevent over-fitting and improve the model’s performance during production.
    """
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
