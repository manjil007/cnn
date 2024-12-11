from Convolution import cnn
from linear import TwoLayerNetwork
import numpy as np
from utils import cross_entropy_loss, softmax
from activation import Activation


class LeNet5:
    def __init__(self, in_channel, lr=0.01):
        self.lr = lr
        self.conv1 = cnn(
            in_channel=in_channel, out_channel=6, kernel_size=5, stride=1, padding=0
        )
        self.relu1 = Activation("relu")

        # Second convolutional layer: 16 filters, 5x5 kernel
        self.conv2 = cnn(
            in_channel=6, out_channel=16, kernel_size=5, stride=1, padding=0
        )
        self.relu2 = Activation("relu")


    def forward(self, x: np.ndarray):
        # Pass through the first convolutional layer
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)

        # Pass through the second convolutional layer
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)

        self.flattened_size = x.shape

        x = x.reshape(x.shape[0], -1)

        self.fc1 = TwoLayerNetwork(x.shape[1], hidden_size=120, out_size=10)
        x = self.fc1.forward(x, self.relu1)

        probabilities = softmax(x)

        return probabilities

    def backward(self, gradient):
        dz = self.fc1.backward(gradient, self.relu1)

        dz = dz.reshape(self.flattened_size)
        dz = self.relu2.backward(dz)

        dz = self.conv2.backward(dz)

        dz = self.conv1.backward(dz)

        return dz
    
    def update_params(self):
        self.conv1.update_params(self.lr)
        self.conv2.update_params(self.lr)

        self.fc1.update_params(self.lr)
