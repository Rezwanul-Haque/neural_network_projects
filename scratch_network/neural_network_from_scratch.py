import numpy as np


# Activation function
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# Activation function
def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    """
    A class of neural network without using any third party libraries
    """

    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    # feed forward network
    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    # back propagation
    def backprop(self):
        # Application of the chain rule to find the derivation of the loss function
        # with respect to weights1 and weights2
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))

        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]
                  ])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y)

    for i in range(1500):
        nn.feed_forward()
        nn.backprop()

    print(nn.output)
