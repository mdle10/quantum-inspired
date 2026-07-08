import matplotlib.pyplot as plt

def plot_prediction(X, Y, predictions):
    plt.scatter(X, Y, label="True data")
    plt.plot(X, predictions, label="Prediction")

    plt.legend()
    plt.savefig("images/regression_demo.png")
    plt.show()