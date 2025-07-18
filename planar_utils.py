import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets


def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary learned by a classification model over a 2D input space.

    Parameters
    -----------
    model : `function`
        A function that takes a 2D numpy array of shape `(n_samples, 2)` and returns predicted labels.
    X : `numpy.ndarray`
        Input features of shape `(2, n_samples)`. Each column represents a data point in 2D.
    y : `numpy.ndarray`
        True labels corresponding to X, shape `(1, n_samples)`.

    Returns
    --------
    None. Displays a matplotlib plot of the decision boundary.
    """

    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.title("Decision Boundary")

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')

    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


def tanh_derivative(A):
    """
    Compute the tanh derivative of A.

    Parameters
    ----------
    A : A scalar or numpy array of any size.

    Returns
    -------
    output : tanh'(A)
    """

    return 1 - A ** 2


def sigmoid(x):
    """
    Compute the sigmoid of x.

    Parameters
    ----------
    x : A scalar or numpy array of any size.

    Returns
    -------
    output : sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    Compute the ReLU (Rectified Linear Unit) of x.

    Parameters
    ----------
    x : A scalar or numpy array of any size.

    Returns
    -------
    output : relu(x)
    """
    return np.maximum(0, x)


def relu_derivative(A):
    """
    Compute the derivative of the ReLU activation.

    Parameters
    ----------
    A : A scalar or numpy array of any size (output of relu).

    Returns
    -------
    output : relu'(A)
    """
    return (A > 0).astype(float)


def load_planar_dataset():
    """
    Returns dataset of flower-like shape.

    Returns
    -------
    X : `np.ndarray`
    Y : `np.ndarray`
    """

    np.random.seed(1)

    m = 400  # number of examples
    N = int(m/2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N*j, N*(j+1))
        t = np.linspace(j*3.12, (j+1)*3.12, N) + np.random.randn(N)*0.2  # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2  # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def load_extra_datasets():
    """
    Generates and returns a collection of synthetic 2D datasets for classification.

    Returns
    -------
    datasets : `dict`    
    """

    np.random.seed(1)

    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)

    datasets = {
        "noisy_circles": noisy_circles,
        "noisy_moons": noisy_moons,
        "blobs": blobs,
        "gaussian_quantiles": gaussian_quantiles,
    }

    return datasets
