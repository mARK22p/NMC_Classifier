import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ENABLE_DISPLAY = False

# ----------------Internal libraries----------------
from splitters import split_data
from loaders import DataLoaderMNIST
from data_perturb import data_perturb_random
from utils import display_utils
from classifiers import NMC
# --------------End Internal libraries--------------

if __name__ == "__main__":
    # dataset loading phase. Creation of training and test set
    filename = "https://github.com/unica-isde/isde/raw/master/data/mnist_data.csv"
    data_loader = DataLoaderMNIST(filename=filename, n_samples=10000)
    x, y = data_loader.load_data()
    xtr, ytr, xts, yts = split_data(x, y, fraction_tr=0.5)

    if ENABLE_DISPLAY:
        display_utils.plot_ten_digits(x, y)
        plt.show()

    # classifier fit and predict phases
    clf = NMC()
    clf.fit(xtr, ytr)  # fit compute centroids for each class

    if ENABLE_DISPLAY:
        # centroids is a matrix with 10 elements with a mean of training samples
        display_utils.plot_ten_digits(clf.centroids, list(range(0, clf.centroids.shape[0])))
        plt.show()

    y_predicted = clf.predict(xts)

    # compute the test error (fraction of samples that are misclassified)
    print("Test error: " + str(np.mean(yts != y_predicted)))

    # perturbation on training data
    perturbation = data_perturb_random.CDataPerturbRandom()
    x_perturbed = perturbation.data_perturbation(xtr[0,:])

    print(x_perturbed)
