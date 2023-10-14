from abc import ABC, abstractmethod


class DataLoader(ABC):
    """
    Abstract interface to implement data loaders
    """
    @abstractmethod
    def load_data(self):
        raise NotImplementedError("load_data not implemented.")
