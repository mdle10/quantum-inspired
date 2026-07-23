from datasets import data_creation, load_sequence, create_windows
from neural_network import (
    initialize_parameters,
    forward
)
from trainer import train
from utils import plot_prediction

data = load_sequence()
X, Y = create_windows(data, 3)

parameters = initialize_parameters(
    input_size=3,
    hidden_size=8,
    output_size=1
)

parameters, losses = train(
    X,
    Y,
    parameters,
    epochs=1000,
    learning_rate=0.5
)


print(losses)
predictions, _ = forward(X, parameters)
print(predictions)
plot_prediction(Y, predictions)