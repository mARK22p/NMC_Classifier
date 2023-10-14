import numpy as np


def split_data(x, y, fraction_tr=0.6):
    '''

    :param x:
    :param y:
    :param fraction_tr:
    :return:
    '''
    n_samples = x.shape[0]
    idx = list(range(0, n_samples))  # [0 1 ... 999]  np.linspace
    np.random.shuffle(idx) # shuffles the array in place, along the first axis of a multi-dimensional array.
    n_training = int(fraction_tr * n_samples)

    idx_tr = idx[:n_training]
    idx_ts = idx[n_training:]

    # training set
    xtr = x[idx_tr, :]
    ytr = y[idx_tr]

    # test set
    xts = x[idx_ts, :]
    yts = y[idx_ts]

    return xtr, ytr, xts, yts
