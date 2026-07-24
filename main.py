from datasets import create_windows, train_test_split, load_csv, normalize, denormalize
from neural_network import (initialize_parameters, forward)
from trainer import train
from utils import plot_prediction, plot_loss

data, minimum, maximum = normalize(load_csv("data/AAPL.csv"))
X, Y = create_windows(data, 3)
X_train, Y_train, X_test, Y_test = train_test_split(X, Y)


parameters = initialize_parameters(
    input_size=3,
    hidden_size=10,
    output_size=1
)

parameters, losses = train(X_train,Y_train, parameters, epochs=1000, learning_rate=0.5)


print(losses)
predictions, _ = forward(X_test, parameters)
predictions = denormalize(predictions, minimum, maximum)
Y_test = denormalize(Y_test, minimum, maximum)
print(predictions)
plot_prediction(Y_test, predictions)
plot_loss(losses)