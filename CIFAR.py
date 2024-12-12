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
train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, download=False, transform=transform)


image, label = train_dataset[10]

image = np.transpose(image.numpy(), (1, 2, 0))

# CIFAR-10 class names
classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

train_images_numpy = train_images_numpy
train_labels_numpy = train_labels_numpy

epochs = 3
model = LeNet5(in_channel=3, lr=0.001)

batch_size = 256


for epoch in range(epochs):
    indices = np.arange(len(train_images_numpy))
    train_images_numpy = train_images_numpy[indices]
    train_labels_numpy = train_labels_numpy[indices]

    num_batches = len(train_images_numpy) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_images = train_images_numpy[start_idx:end_idx]
        batch_labels = train_labels_numpy[start_idx:end_idx]

        logit = model.forward(batch_images)

        predictions = np.argmax(logit)

        Y = one_hot_y(batch_labels, 10)

        loss = cross_entropy_loss(logit, Y)

        # Print loss for the current batch
        print(f"  Batch {i + 1}/{num_batches}, Loss: {loss:.4f}")

        gradient = logit - Y

        model.backward(gradient)

        model.update_params()

        logit = model.forward(test_images_numpy)

        predictions = np.argmax(logit)

        correct_predictions = np.sum(predictions == test_labels_numpy)

        accuracy = correct_predictions / len(test_labels_numpy)

        print(f"Accuracy: {accuracy * 100:.2f}%")


logit = model.forward(test_images_numpy)

predictions = np.argmax(logit)

correct_predictions = np.sum(predictions == test_labels_numpy)

accuracy = correct_predictions / len(test_labels_numpy)

print(f"Accuracy: {accuracy * 100:.2f}%")

















