import numpy as np


def softmax(a):
    """Softmax"""
    norm_a = np.exp(a - np.max(a, axis=1, keepdims=True))
    return norm_a / np.sum(norm_a, axis=1, keepdims=True)


def cross_entropy_loss(x_pred, y):
    """
    Compute Cross-Entropy Loss based on prediction of the network and labels
    :param x_pred: Probabilities from the model (N, num_classes)
    :param y: Labels of instances in the batch
    :return: The computed Cross-Entropy Loss
    """
    loss = 0
    small_num = 1e-12
    x_pred = np.clip(x_pred, small_num, 1 - small_num)
    for i, v in enumerate(x_pred):
        loss -= np.log(v[y[i]])

    return np.sum(loss) / len(y)


def one_hot_y(Y, num_classes):
    """
    One hot encoding representation
    """
    res = []
    for V in Y:
        temp = [0] * num_classes
        temp[V] = 1
        res.append(temp)
    return np.asarray(res)



def compute_accuracy(x_pred, y):
        """
        Compute the accuracy of current batch
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        """
        correct = 0
        for i, v in enumerate(x_pred):
            index = np.argmax(v)
            if index == y[i]:
                correct += 1

        acc = correct / len(y)
        return acc
