import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ENABLE_DISPLAY = False

# ----------------Internal libraries----------------
from splitters import split_data
from loaders import CDataLoaderMNIST
from data_perturb import CDataPerturbRandom
from utils import display_utils
from classifiers import NMC
# --------------End Internal libraries--------------

if __name__ == "__main__":
    # dataset loading phase. Creation of training and test set
    filename = "https://github.com/unica-isde/isde/raw/master/data/mnist_data.csv"
    data_loader = CDataLoaderMNIST(filename = filename, n_samples = 10000, normalize = False)
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
    
    # perturbation on training data
    perturbation = CDataPerturbRandom(k=100)
    img = xtr[0,:]
    plt.figure()
    plt.imshow(img.reshape(28, 28), cmap="gray")
    plt.show()
    img_perturbed = perturbation.data_perturbation(img)
    plt.figure()
    plt.imshow(img_perturbed.reshape(28, 28), cmap="gray")
    plt.show()
    