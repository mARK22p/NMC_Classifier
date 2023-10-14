import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from splitters import split_data
from loaders import DataLoaderMNIST
from data_perturb import data_perturb_random
from utils import display_utils

from classifiers import NMC

filename = "https://github.com/unica-isde/isde/raw/master/data/mnist_data.csv"

data_loader = DataLoaderMNIST(filename=filename, n_samples=10000)
clf = NMC()

x, y = data_loader.load_data()
plot_ten_digits(x, y)
plt.show()

xtr, ytr, xts, yts = split_data(x, y, fraction_tr=0.5)

clf.fit(xtr, ytr)
ypred = clf.predict(xts)
# plot_ten_digits(clf.centroids)
# plt.show()

# compute the test error (fraction of samples that are misclassified)
print("Test error: " + str(np.mean(yts != ypred)))


