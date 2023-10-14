import numpy as np
import pandas as pd

from .data_loader import DataLoader

class DataLoaderMNIST(DataLoader):
    def __init__(self, filename, n_samples = None):
        self._filename = filename
        self._n_samples = n_samples

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, new_filename):
        self._filename = new_filename

    def load_data(self):
        data = pd.read_csv(self._filename) # reading data from the CSV file into a pandas DataFrame
        data = np.array(data) # construct a ndarray from a pandas DataFrame




