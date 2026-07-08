from datasets import data_creation
from neural_network import (
    initialize_parameters,
    forward
)
from trainer import train
from utils import plot_prediction

X, Y = data_creation()

parameters = initialize_parameters(
    input_size=1,
    hidden_size=8,
    output_size=1
)

parameters, losses = train(
    X,
    Y,
    parameters,
    epochs=10000,
    learning_rate=0.5
)

print(losses)
predictions, _ = forward(X, parameters)

plot_prediction(X, Y, predictions)