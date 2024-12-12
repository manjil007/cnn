import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
from CNN import LeNet5
from utils import cross_entropy_loss, one_hot_y, compute_accuracy, softmax, plot_loss_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)


image, label = train_dataset[10]

image = np.transpose(image.numpy(), (1, 2, 0))

train_images_tensor = torch.stack(
    [train_dataset[i][0] for i in range(len(train_dataset))]
)
train_labels_tensor = torch.tensor(
    [train_dataset[i][1] for i in range(len(train_dataset))]
)

test_images_tensor = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
test_labels_tensor = torch.tensor(
    [test_dataset[i][1] for i in range(len(test_dataset))]
)

train_images_numpy = train_images_tensor.numpy()
train_labels_numpy = train_labels_tensor.numpy()

test_images_numpy = test_images_tensor.numpy()
test_labels_numpy = test_labels_tensor.numpy()

train_images_numpy = train_images_numpy[:6000]
train_labels_numpy = train_labels_numpy[:6000]


epochs = 10
model = LeNet5(in_channel=1, lr=1e-1)

batch_size = 120
losses = []


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

        logits = model.forward(batch_images)
        probabilities = softmax(logits)

        Y = one_hot_y(batch_labels, 10)

        loss = cross_entropy_loss(probabilities, Y)
        total_loss += loss

        gradient = probabilities - Y

        model.backward(gradient)

        model.update_params()

        model.zero_gradient()

        prob = model.forward(batch_images)
        x_pred = np.argmax(prob, axis=1)
        accuracy = compute_accuracy(x_pred, batch_labels)

    print(f"Epoch : {epoch}, loss {total_loss/num_batches}, accuracy: {accuracy}")
    losses.append(total_loss)


prob = model.forward(test_images_numpy)
x_pred = np.argmax(prob, axis=1)
accuracy = compute_accuracy(x_pred, test_labels_numpy)
print(f"Accuracy: {accuracy * 100:.2f}%")

loss_plot = plot_loss_curve(epochs=len(losses), losses=losses)

loss_plot.savefig('./plots/loss_curve_mnist.png')


cm = confusion_matrix(test_labels_numpy, x_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('./plots/confusion_mnist.png')

