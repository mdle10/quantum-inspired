import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))


def initialize_parameters():

    np.random.seed(42)

    parameters = {
        "W1" : np.random.randn(2,5) * 0.5,
        "b1" : np.zeros((1,5)),

        "W2" : np.random.randn(5,1) * 0.5,
        "b2": np.zeros((1,1))
    }

    return parameters

def forward(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]

    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = (X @ W1) + b1
    A1 = sigmoid(Z1)
    Z2 = (A1 @ W2) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache
def compute_loss(Y, A2):

    return np.mean((Y - A2)**2) / 2

def backward(X, Y, parameters, cache):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    delta2 = (A2 - Y) * A2 * (1 - A2)

    m = X.shape[0]

    dW2 = (A1.T @ delta2)/m
    db2 = (np.sum(delta2, axis=0, keepdims=True))/m
    delta1 = (delta2 @ W2.T) * A1 * (1 - A1)
    dW1 = (X.T @ delta1)/m
    db1 = (np.sum(delta1, axis=0, keepdims=True))/m

    gradients ={
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return gradients
parameters = initialize_parameters()

def update(parameters, gradients):
    learning_rate = 0.5

    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]

    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]

    return parameters

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])
for epoch in range(90000):

    A2, cache = forward(X, parameters)
    print(A2)

    loss = compute_loss(Y, A2)

    gradients = backward(X, Y, parameters, cache)


    parameters = update(parameters, gradients)
    print(epoch, loss)
