import numpy as np
import pandas as pd

def load_csv(filepath, column="Close/Last"):
    df = pd.read_csv(filepath)
    data = df["Close"]
    return data

def load_sequence():
    x = np.linspace(0, 10 * np.pi, 500)
    data = np.sin(x)
    return data

#Regression Demo:
def data_creation():
    X = np.linspace(-2, 2, 200).reshape(-1, 1)
    Y = X ** 2
    return X, Y

def create_windows(data, window_size):
    X = []
    Y = []
    i = 0

    while i < len(data) - window_size:
        X.append(data[i:i+window_size])
        Y.append(data[i+window_size])
        i += 1

    return np.array(X), np.array(Y).reshape(-1,1)

def normalize(data):
    minimum = np.min(data)
    maximum = np.max(data)

    return ((data - minimum) / (maximum - minimum)), minimum, maximum

def denormalize(data, minimum, maximum):
    return (data*(maximum - minimum)) + minimum

def train_test_split(X, Y):
    split = int(0.8*len(X))

    X_train = X[:split]
    X_test = X[split:]
    Y_train = Y[:split]
    Y_test = Y[split:]
    return X_train, Y_train, X_test, Y_test