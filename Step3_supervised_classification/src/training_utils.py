import torch
import torch.nn as nn
from collections import Counter


class State:
    """
    class used to save models during training
    """

    def __init__(self, model, optim) -> None:
        self.model = model
        self.optimizer = optim
        self.epoch = 0


def weight1(train_dataset, class_nb):
    """
    First way to define weights to apply to loss in order to leverage unbalanced data
    """
    y_train = []
    for x, y, _ in train_dataset:
        for i in range(x.shape[0]):
            y_train.append(y[i].item())
    count = Counter(y_train)
    weight = []
    for i in range(class_nb):
        weight.append(1 - count[i] / len(y_train))

    return weight


def weight2(train_dataset, class_nb):
    """
    Second way to define weights to apply to loss in order to leverage unbalanced data
    """
    y_train = []
    for x, y, _ in train_dataset:
        for i in range(x.shape[0]):
            y_train.append(y[i].item())
    count = Counter(y_train)
    weight = []
    for i in range(class_nb):
        weight.append(len(y_train) / count[i])

    return weight


def reset_weights(m):
    """
    Initializes the weights of all the layers of a model
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()


def init_weights_xavier(m):
    """
    Xavier initialization
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def init_weights_kaining(m, nonlinearity="ReLu"):
    """
    Kaiming initialization
    """
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        m.bias.data.fill_(0.01)
