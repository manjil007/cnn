import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision

import gc

import os
import sys
from cnn import LeNet5
from utils import cross_entropy_loss, one_hot_y


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)


image, label = train_dataset[10]

image = np.transpose(image.numpy(), (1, 2, 0))

train_images_tensor = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
train_labels_tensor = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

test_images_tensor = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
test_labels_tensor = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

train_images_numpy = train_images_tensor.numpy()
train_labels_numpy = train_labels_tensor.numpy()

test_images_numpy = test_images_tensor.numpy()
test_labels_numpy = test_labels_tensor.numpy()

test_images_numpy = test_images_numpy[:50]
test_labels_numpy = test_labels_numpy[:50]

train_images_numpy = train_images_numpy[:500]
train_labels_numpy = train_labels_numpy[:500]

epochs = 4
model = LeNet5(in_channel=1, lr=0.001)

batch_size = 32


for epoch in range(epochs):
    indices = np.arange(len(train_images_numpy))
    train_images_numpy = train_images_numpy[indices]
    train_labels_numpy = train_labels_numpy[indices]

    num_batches = len(train_images_numpy) // batch_size
    total_loss = 0
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_images = train_images_numpy[start_idx:end_idx]
        batch_labels = train_labels_numpy[start_idx:end_idx]

        probabilities = model.forward(batch_images)
        
        Y = one_hot_y(batch_labels, 10)

        loss = cross_entropy_loss(probabilities, Y)
        total_loss += loss

        # Print loss for the current batch
        print(f"  Batch {i + 1}/{num_batches}, Loss: {loss:.4f}")

        gradient = probabilities - Y

        model.backward(gradient)

        model.update_params()
        
        model.zero_gradient()

        probabilities = model.forward(batch_images)
        predictions = np.argmax(probabilities)
        correct_predictions = np.sum(predictions == batch_labels)
        accuracy = correct_predictions / len(batch_labels
                                             )

        print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Epoch : {epoch}, loss {total_loss/num_batches}")



probabilities = model.forward(test_images_numpy)

predictions = np.argmax(probabilities)

correct_predictions = np.sum(predictions == test_labels_numpy)

accuracy = correct_predictions / len(test_labels_numpy)

print(f"Accuracy: {accuracy * 100:.2f}%")

















