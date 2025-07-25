import numpy as np
from planar_utils import sigmoid, tanh_derivative, relu, relu_derivative

np.random.seed(1)


def layer_sizes(X, Y, hidden_units_1=32, hidden_units_2=32):
    """
    Computes the sizes of the layers in a 3-layer neural network.

    Parameters
    ----------
    X : `np.ndarray`
        Input data of shape `(n_x, m)`, where n_x is the number of features and m is the number of examples.

    Y : `np.ndarray`
        Input data of shape `(n_y, m)`, where n_y is the number of output units and m is the number of examples.

    hidden_units_1 : `int, optional`
        Number of neurons in the 1st hidden layer. Default is 32.

    hidden_units_2 : `int, optional`
        Number of neurons in the 2nd hidden layer. Default is 32.

    Returns
    -------
    tuple
        A tuple (n_x, n_h, n_y) where:
        - n_x : `int`. Number of input features (size of input layer).
        - n_h_1 : `int`. Number of hidden units in 1st layer.
        - n_h_2 : `int`. Number of hidden units in 2nd layer.
        - n_y : `int`. Number of output units (size of output layer).
    """

    n_x = X.shape[0]
    n_h_1 = hidden_units_1
    n_h_2 = hidden_units_2
    n_y = Y.shape[0]

    return (n_x, n_h_1, n_h_2, n_y)


def initialize_parameters(n_x, n_h_1, n_h_2, n_y):
    """
    Initializes the parameters of a 3-layer neural network.

    Parameters
    ----------
    n_x : `int`
        Number of input features (input layer size).
    n_h_1 : `int`
        Number of neurons in the 1st hidden layer.
    n_h_2 : `int`
        Number of neurons in the 2nd hidden layer.
    n_y : `int`
        Number of output units (output layer size).

    Returns
    --------
    parameters : `dict`
        A dictionary containing:
        - W1 : Weight matrix of shape `(n_x, n_h_1)`
        - b1 : Bias vector of shape `(n_h_1, 1)`
        - W2 : Weight matrix of shape `(n_h_1, n_h_2)`
        - b2 : Bias vector of shape `(n_h_2, 1)`
        - W3 : Weight matrix of shape `(n_h_2, n_y)`
        - b3 : Bias vector of shape `(n_y, 1)`
    """

    # If you need to generate the same sequence of random numbers multiple times within a program, you'll need to set the seed again before each generation.
    np.random.seed(1)

    parameters = {
        "W1": np.random.randn(n_x, n_h_1) * (np.sqrt(1. / n_h_1)),
        "b1": np.zeros((n_h_1, 1)),
        "W2": np.random.randn(n_h_1, n_h_2) * (np.sqrt(1. / n_h_2)),
        "b2": np.zeros((n_h_2, 1)),
        "W3": np.random.randn(n_h_2, n_y) * (np.sqrt(1. / n_y)),
        "b3": np.zeros((n_y, 1))
    }

    return parameters


def forward_propagation(X: np.ndarray, parameters: dict, activation: str = "tanh") -> dict:
    """
    Implements forward propagation for a 3-layer neural network.

    Parameters
    ----------
    parameters : `dict`
        Dictionary containing the parameters:
        - W1 : `np.ndarray` of shape `(n_x, n_h_1)`.
            Weight matrix for the first layer
        - b1 : `np.ndarray` of shape `(n_h_1, 1)`.
            Bias vector for the first layer
        - W2 : `np.ndarray` of shape `(n_h_1, n_h_2)`.
            Weight matrix for the second layer
        - b2 : `np.ndarray` of shape `(n_h_2, 1)`.
            Bias vector for the second layer
        - W3 : `np.ndarray` of shape `(n_h_2, n_y)`.
            Weight matrix for the second layer
        - b3 : `np.ndarray` of shape `(n_y, 1)`.
            Bias vector for the second layer
        - activation : `str`
            Activation function used in the hidden layers. Options are "tanh", "relu", or "sigmoid".

    X : `np.ndarray`
        Input data of shape `(n_x, m)`, where n_x is the number of features and m is the number of examples.

    Returns
    -------
    cache : `dict`
        Dictionary containing intermediate values:
        - Z1 : `np.ndarray`.
            Linear activation of first layer
        - A1 : `np.ndarray`.
            Activation from first layer (tanh)
        - Z2 : `np.ndarray`.
            Linear activation of second layer
        - A2 : `np.ndarray`.
            Activation from second layer (tanh)
        - Z3 : `np.ndarray`.
            Linear activation of third layer
        - A3 : `np.ndarray`.
            Activation from third layer (sigmoid)
    """

    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = np.dot(W1.T, X) + b1
    if activation == "relu":
        A1 = relu(Z1)
    elif activation == "sigmoid":
        A1 = sigmoid(Z1)
    else:
        A1 = np.tanh(Z1)

    Z2 = np.dot(W2.T, A1) + b2
    if activation == "relu":
        A2 = relu(Z2)
    elif activation == "sigmoid":
        A2 = sigmoid(Z2)
    else:
        A2 = np.tanh(Z2)

    Z3 = np.dot(W3.T, A2) + b3
    A3 = sigmoid(Z3)

    cache = {
        "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2,
        "Z3": Z3, "A3": A3
    }

    return cache


def backward_propagation(parameters: dict, cache: dict, X: np.ndarray, Y: np.ndarray, activation: str = "tanh") -> dict:
    """
    Implements backward propagation for a 3-layer neural network.

    Parameters
    ----------
    parameters : `dict`
        Dictionary containing the parameters:
        - W1 : `np.ndarray` of shape `(n_x, n_h_1)`
            Weight matrix for the first layer
        - b1 : `np.ndarray` of shape `(n_h_1, 1)`
            Bias vector for the first layer
        - W2 : `np.ndarray` of shape `(n_h_1, n_h_2)`
            Weight matrix for the second layer
        - b2 : `np.ndarray` of shape `(n_h_2, 1)`
            Bias vector for the second layer
        - W3 : `np.ndarray` of shape `(n_h_2, n_y)`
            Weight matrix for the third layer
        - b3 : `np.ndarray` of shape `(n_y, 1)`
            Bias vector for the third layer
        - activation : `str`
            Activation function used in the hidden layers. Options are "tanh", "relu", or "sigmoid".

    cache : `dict`
        Dictionary containing intermediate values:
        - Z1 : `np.ndarray`.
            Linear activation of first layer
        - A1 : `np.ndarray`.
            Activation from first layer (tanh)
        - Z2 : `np.ndarray`.
            Linear activation of second layer
        - A2 : `np.ndarray`.
            Activation from second layer (tanh)
        - Z3 : `np.ndarray`.
            Linear activation of third layer
        - A3 : `np.ndarray`.
            Activation from third layer (sigmoid)

    X : `np.ndarray`
        Input data of shape `(n_x, m)`, where n_x is the number of features and m is the number of examples.

    Y : `np.ndarray`
        Input data of shape `(n_y, m)`, where n_y is the number of output units and m is the number of examples.

    Returns
    -------
    grads : dict
        Dictionary containing gradients of the cost with respect to parameters:
        - dW1 : `np.ndarray`
        - db1 : `np.ndarray`
        - dW2 : `np.ndarray`
        - db2 : `np.ndarray`
        - dW3 : `np.ndarray`
        - db3 : `np.ndarray`
    """

    m = X.shape[1]

    W3 = parameters["W3"]
    W2 = parameters["W2"]

    A1 = cache.get("A1")
    A2 = cache.get("A2")
    A3 = cache.get("A3")

    dZ3 = A3 - Y
    dW3 = np.dot(A2, dZ3.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    if activation == "relu":
        dZ2 = np.dot(W3, dZ3) * relu_derivative(A2)
    elif activation == "sigmoid":
        dZ2 = np.dot(W3, dZ3) * (A2 * (1 - A2))
    else:
        dZ2 = np.dot(W3, dZ3) * tanh_derivative(A2)

    dW2 = np.dot(A1, dZ2.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    if activation == "relu":
        dZ1 = np.dot(W2, dZ2) * relu_derivative(A1)
    elif activation == "sigmoid":
        dZ1 = np.dot(W2, dZ2) * (A1 * (1 - A1))
    else:
        dZ1 = np.dot(W2, dZ2) * tanh_derivative(A1)

    dW1 = np.dot(X, dZ1.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2,
        "dW3": dW3, "db3": db3
    }

    return grads


def compute_cost(A3: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the cost for the 3-layer neural network.

    Parameters
    ----------
    A3 : `np.ndarray`
        Activation from third layer (sigmoid).
    Y : `np.ndarray`
        Input data containing true labels of shape `(n_y, m)`, where n_y is the number of output units and m is the number of examples.

    Returns
    -------
    cost : `float`
        The computed cost value.
    """

    m = Y.shape[1]
    epsilon = 1e-15

    cost = -(np.sum(((Y * np.log(A3 + epsilon)) + (1 - Y) * np.log(1 - A3 + epsilon)))) / m
    cost = np.squeeze(cost)

    return cost


def update_parameters(parameters: dict, grads: dict, learning_rate: float) -> dict:
    """
    Updates the parameters of a 3-layer neural network using gradient descent.

    Parameters
    ----------
    parameters : `dict`
        Dictionary containing current parameters:
        - W1 : `np.ndarray` of shape `(n_x, n_h_1)`
            Weights for the first layer
        - b1 : `np.ndarray` of shape `(n_h_1, 1)`
            Biases for the first layer
        - W2 : `np.ndarray` of shape `(n_h_1, n_h_2)`
            Weights for the second layer
        - b2 : `np.ndarray` of shape `(n_h_2, 1)`
            Biases for the second layer
        - W3 : `np.ndarray` of shape `(n_h_2, n_y)`
            Weights for the third layer
        - b3 : `np.ndarray` of shape `(n_y, 1)`
            Biases for the third layer

    grads : `dict`
        Dictionary containing gradients of the cost with respect to parameters:
        - dW1 : `np.ndarray`
        - db1 : `np.ndarray`
        - dW2 : `np.ndarray`
        - db2 : `np.ndarray`
        - dW3 : `np.ndarray`
        - db3 : `np.ndarray`

    learning_rate : `float`
        The learning rate used to update the parameters.

    Returns
    -------
    parameters : `dict`
        Dictionary containing updated parameters:
        - W1, b1, W2, b2, W3, b3 (same shapes as input parameters)
    """

    updated_params = {}

    for l in range(1, 4):  # for 3 layers
        W_key = f"W{l}"
        b_key = f"b{l}"
        dW_key = f"dW{l}"
        db_key = f"db{l}"

        updated_params[W_key] = parameters[W_key] - learning_rate * grads[dW_key]
        updated_params[b_key] = parameters[b_key] - learning_rate * grads[db_key]

    return updated_params


def build_model(
    X: np.ndarray,
    Y: np.ndarray,
    hidden_units_1: int = 16,
    hidden_units_2: int = 16,
    learning_rate: float = 0.01,
    num_iterations: int = 5000,
    activation: str = "tanh"
) -> dict:
    """
    Builds and trains the model for a 3-layer neural network.

    Parameters
    ----------
    X : `np.ndarray`
        Input data of shape `(n_x, m)`, where n_x is the number of features and m is the number of examples.

    Y : `np.ndarray`
        Input data of shape `(n_y, m)`, where n_y is the number of output units and m is the number of examples.

    hidden_units : `int, optional`
        Number of neurons in the hidden 2nd layer. Default is 16.

    learning_rate : `float, optional`
        The learning rate for the update step. Default is 0.01.

    num_iterations : `int, optional`
        Number of iterations for training the model. Default is 5000.

    activation : `str`
        Activation function used in the hidden layers. Options are "tanh", "relu", or "sigmoid".

    Returns
    -------
    parameters : `dict`
        Updated parameters after applying the gradients. Received from the `update_parameters` function.
    """

    n_x, n_h_1, n_h_2, n_y = layer_sizes(X, Y, hidden_units_1, hidden_units_2)
    parameters = initialize_parameters(n_x, n_h_1, n_h_2, n_y)

    for i in range(0, num_iterations):
        cache = forward_propagation(X, parameters, activation)
        grads = backward_propagation(parameters, cache, X, Y, activation)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 50 == 0:
            A3 = cache.get("A3")
            cost = compute_cost(A3, Y)
            print(f"Cost after iteration {i}: {cost:.6f}")

    return parameters


def predict(parameters: dict, X: np.ndarray) -> np.ndarray:
    """
    Predicts binary labels for the input data through a 3-layer neural network.

    Parameters
    ----------
    parameters : dict
        Dictionary containing the trained parameters:
        - W1 : `np.ndarray` of shape `(n_x, n_h_1)`
            Weight matrix for the first layer
        - b1 : `np.ndarray` of shape `(n_h_1, 1)`
            Bias vector for the first layer
        - W2 : `np.ndarray` of shape `(n_h_1, n_h_2)`
            Weight matrix for the second layer
        - b2 : `np.ndarray` of shape `(n_h_2, 1)`
            Bias vector for the second layer
        - W3 : `np.ndarray` of shape `(n_h_2, n_y)`
            Weight matrix for the third layer
        - b3 : `np.ndarray` of shape `(n_y, 1)`
            Bias vector for the third layer

    X : np.ndarray
        Input data of shape `(n_x, m)`, where n_x is the number of input features and m is the number of examples.

    Returns
    -------
    A3 : np.ndarray
        Binary predictions of shape `(1, m)`, where each value is 0 or 1.
    """

    cache = forward_propagation(X, parameters)
    A3 = (cache["A3"] > 0.5).astype(int)

    return A3
