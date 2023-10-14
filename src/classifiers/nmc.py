import numpy as np


class NMC:
    """
    Nearest Mean Centroid (NMC) Classifier
    ...
    """

    def __init__(self):
        self._centroids = None

    # centroids are read only
    @property
    def centroids(self):
        return self._centroids

    def fit(self, xtr, ytr):
        n_classes = np.unique(ytr).size  # compute how many classes there are in the dataset
        n_features = xtr.shape[1]

        self._centroids = np.zeros(shape=(n_classes, n_features))

        ''' For loop instruction explanation:
            The next instruction assigns to self._centroids the mean of every image present in the dataset. 
            Xtr[ytr==k,:] generates a boolean vector useful for selecting from the training set only the images of the 
            same class, denoted by k. The axis=0 parameter is used to compute the mean along the row axis (where 
            axis=0 refers to the row axis). The returned value is a row representing the mean values for each class 
        '''
        for k in range(0, n_classes):

            self._centroids[k, :] = np.mean(xtr[ytr == k, :], axis=0)

    def predict(self, xts):
        n_samples = xts.shape[0]
        n_classes = self._centroids.shape[0]

        dist = np.zeros(shape=(n_samples, n_classes))

        ''' For loop instruction explanation:
            xts - self._centroids[k,:] compute the difference above test set and k-esim centroid. In this phase, 
            xts has a different shape than self._centroids[k,:]. For instance, for each iteration, self._centroids[k,:]
            has a shape of (1,784). There is a broadcasting operation caused by _centroids's shape,
            because the vector is expanded over the rows to get the same length of xts's rows. 
            dist[:,k] contains in k-column the value of each image subtracted to k-esim centroid.
            The result is squared (based on the Euclidian's norm), then summed  in row axis (axis = 1).
            dist.shape = (n_samples,n_classes)
        '''
        for k in range(0, n_classes):
            dist[:, k] = np.sum((xts - self._centroids[k, :]) ** 2, axis=1)

        #  get index of the minimum values along an axis. y_predicted.shape = (n_samples,)
        y_predicted = np.argmin(dist, axis=1)

        return y_predicted
