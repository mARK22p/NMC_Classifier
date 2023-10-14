from abc import ABC, abstractmethod

class CDataPerturb(ABC):

    def __init__(self, X):
        self._X = X

    @abstractmethod
    def data_perturbation(self, x):
        raise NotImplementedError("You must implement data_pertubation in child classes")

    def perturb_dataset(self):
        for i in range(0):
            pass
            #data_perturbation(_X[i])