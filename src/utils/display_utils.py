import matplotlib.pyplot as plt


def plot_ten_digits(x, y=None, shape=(28, 28)):
    for i in range(0, 10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i, :].reshape(shape[0], shape[1]), cmap='gray')
        if y is not None:
            plt.title('Label: ' + str(y[i]))
