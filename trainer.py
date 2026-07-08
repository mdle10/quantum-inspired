import numpy as np
from neural_network import(
    initialize_parameters,
    forward,
    compute_loss,
    backward,
    update

)
parameters = initialize_parameters(
    input_size = 1,
    hidden_size = 3,
    output_size= 1
)

from datasets import data_creation

X, Y = data_creation()

def train(X, Y, parameters, epochs, learning_rate):
    losses = []

    for epoch in range(9000):
        A2, cache = forward(X, parameters)

        loss = compute_loss(Y, A2)

        gradients = backward(X, Y, parameters, cache)

        parameters = update(parameters, gradients, learning_rate)

        losses.append(float(loss))
    return parameters, losses