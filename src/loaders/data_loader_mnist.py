from loaders import CDataLoader
import pandas as pd
import numpy as np


class CDataLoaderMNIST(CDataLoader):

    def __init__(self, filename, n_samples=None):
        self.filename = filename
        self.n_samples = n_samples

    def load_data(self):
        '''
            Mnist dataset contains in the first column the labels. Images are from the second column to row-end
            :return:
        '''
        data = pd.read_csv(self.filename)
        data = np.array(data)  # cast pandas dataframe to numpy array
        if self.n_samples is not None:
            data = data[:self.n_samples, :] # take only the first n_samples rows
        x = data[:, 1:] / 255 # take every row from data matrix, starting from second column. x is a matrix
        y = data[:, 0] # take the first column from data matrix. y is a list
        return x, y
