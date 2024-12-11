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

        self.fc1 = TwoLayerNetwork(6400, hidden_size=120, out_size=10)

    def forward(self, x: np.ndarray):
        # Pass through the first convolutional layer
        conv1_out = self.conv1.forward(x)
        self.conv1_out = conv1_out
        relu1_out = self.relu1.forward(conv1_out)

        # Pass through the second convolutional layer
        conv2_out = self.conv2.forward(relu1_out)
        self.conv2_out = conv2_out 
        relu2_out = self.relu2.forward(conv2_out)

        self.flattened_size = relu2_out.shape

        flat_x  = np.array([relu2_out[s].flatten() for s in range(relu2_out.shape[0])])

        fc1_out = self.fc1.forward(flat_x, self.relu1)
        probabilities = softmax(fc1_out)
        return probabilities

    def backward(self, gradient):
        dz = self.fc1.backward(gradient, self.relu1)
        
        dz = np.array([ dz[i].reshape(self.flattened_size[1], self.flattened_size[2], self.flattened_size[3]) for  i in range(self.flattened_size[0])])

        relu2_der = self.relu2.backward(self.conv2_out)
        dz = dz * relu2_der

        dz = self.conv2.backward(dz)
        relu1_der = self.relu1.backward(self.conv1_out)
        dz = dz * relu1_der

        dz = self.conv1.backward(dz)

        return 
    
    def update_params(self):
        self.conv1.update_params(self.lr)
        self.conv2.update_params(self.lr)

        self.fc1.update_params(self.lr)
        
    def zero_gradient(self):
        self.conv1.zero_gradient()
        self.conv2.zero_gradient()
        self.fc1.zero_gradient()
