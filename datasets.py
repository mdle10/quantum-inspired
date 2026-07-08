import numpy as np

def data_creation():
    X = np.linspace(-2, 2, 200).reshape(-1, 1)
    Y = X ** 2
    return X, Y