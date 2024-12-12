from Convolution import cnn
from linear import TwoLayerNetwork
import numpy as np
from utils import cross_entropy_loss, softmax
from relu import ReLU


class LeNet5:
    def __init__(self, in_channel, lr=1e-4):
        self.lr = lr
        self.conv1 = cnn(
            in_channel=in_channel, out_channel=6, kernel_size=5, stride=1, padding=0
        )
        self.relu1 = ReLU()

        # Second convolutional layer: 16 filters, 5x5 kernel
        self.conv2 = cnn(
            in_channel=6, out_channel=16, kernel_size=5, stride=1, padding=0
        )
        self.relu2 = ReLU()
        self.fcrelu = ReLU()

        self.fc1 = TwoLayerNetwork(9216, hidden_size=120, out_size=10)

    def forward(self, x: np.ndarray):
        # Pass through the first convolutional layer
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)  

        # Pass through the second convolutional layer
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)

        self.flattened_size = x.shape

        flat_x  = np.array([x[s].flatten() for s in range(x.shape[0])])

        fc1_out = self.fc1.forward(flat_x, self.fcrelu)

        return fc1_out

    def backward(self, gradient):

        dz = self.fc1.backward(gradient, self.fcrelu)
        dz = np.array([ dz[i].reshape(self.flattened_size[1], self.flattened_size[2], self.flattened_size[3]) for  i in range(self.flattened_size[0])])

        dz = self.relu2.backward(dz)
        dz = self.conv2.backward(dz)

        dz = self.relu1.backward(dz) 
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
