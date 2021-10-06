import pandas as pd
import numpy as np
import torch
import math

from torch import nn
from torch.utils.data import DataLoader, TensorDataset, dataloader, random_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt


def get_data():
    # Read Data
    df = pd.read_csv("wine.csv")
    y = df.values[:, 0]
    X = df.values[:, 1:]

    # Scale btw 0 and 1
    scaler = MinMaxScaler()
    Xnorm = scaler.fit_transform(X)

    # Create tensors
    yTensor = torch.from_numpy(y.astype(int) - 1)
    XTensor = torch.from_numpy(X)

    # One hot encoding
    yEnc = nn.functional.one_hot(yTensor, num_classes=3)

    return XTensor, yTensor


def create_datasets(XTensor, yTensor, trainSize):
    ds = TensorDataset(XTensor, yTensor)
    numTrain = math.ceil(len(yTensor) * trainSize)
    numTest = len(yTensor) - numTrain
    dsTrain, dsTest = random_split(ds, [numTrain, numTest])
    return dsTrain, dsTest


def get_dataloaders(dsTrain, dsTest, batchSize):
    batchSize = 64
    dlTrain = DataLoader(dsTrain, batchSize, shuffle=True)
    dlTest = DataLoader(dsTest, batchSize)
    return dlTrain, dlTest

def plot_loss(training_loss, validation_loss):
    plt.title('Training & Validation Error vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(training_loss, label='Training Error')
    plt.plot(validation_loss, label='Validation Error')
    plt.legend()
    plt.show()

def plot_accuracy(accuracy):
    plt.title('Validation Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracy, label='Accuracy')
    plt.legend()
    plt.show()

