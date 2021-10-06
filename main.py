import math
import torch
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data.dataset import random_split
from sklearn.metrics import confusion_matrix

from neural_network import NeuralNetwork, train, test
from helper_functions import get_data, create_datasets, get_dataloaders
from helper_functions import plot_loss, plot_accuracy


def single_run(stack, learning_rate, stopping_criteria, returnVals):
    # Init model
    model = NeuralNetwork(stack)
    print(model)

    # exit()
    # CSV data as tensor objects
    X, y = get_data()

    # Tensor data set objects that are split for train, validation, and test
    trainSize = .8
    dsTrain, dsTest = create_datasets(X, y, trainSize)

    numTrainB = math.ceil(len(dsTrain)*.9)
    numVal = len(dsTrain) - numTrainB
    dsTrainB, dsVal = random_split(dsTrain, [numTrainB, numVal])

    # Data Loader objects that are use in training and testing, describe batchSize
    batchSize = 32
    dlTrain, dlTest = get_dataloaders(dsTrain, dsTest, 1)
    dlTrainB, dlVal = get_dataloaders(dsTrainB, dsVal, batchSize)

    # Model Paramaters
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 1000

    # Train and Plot Train Loss / Validation Loss / Validation Accuracy
    countEpochsTrain, training_loss, validation_loss, validation_accuracy, predictions, true = train(
        dlTrainB, dlVal, model, loss_fn, optimizer, epochs, stopping_criteria, 100)
    confMatVal = confusion_matrix(true, predictions)
    print('Number of Epochs to Train = {}'.format(countEpochsTrain))
    print('Final Training Loss = {}'.format(training_loss[-1]))
    print('Validation Loss = {}'.format(validation_loss[-1]))
    print('Validation Accuracy = {}'.format(validation_accuracy[-1]))
    print('Validation Confusion Matrix\n{}'.format(str(confMatVal)))

    plot_loss(training_loss, validation_loss)
    plot_accuracy(validation_accuracy)

    # Retrain with all data and Test, no stopping criteria but same number of epochs
    test_loss, test_accuracy, predictions, true = test(
        dlTest, model, loss_fn, False)

    # print(predictions,  true)
    confMatTest = confusion_matrix(true, predictions)
    print('Test Accuracy = {}'.format(test_accuracy))
    print('Test Confusion Matrix\n{}'.format(str(confMatTest)))

    if returnVals == True:
        return [countEpochsTrain, training_loss[-1], validation_loss[-1], validation_accuracy[-1], test_accuracy]


hidden_layers = [
    [nn.Linear(13, 12), nn.ReLU(), nn.Linear(12, 3)],
    [nn.Linear(13, 12), nn.ReLU(), nn.Linear(
        12, 6), nn.ReLU(), nn.Linear(6, 3)],
    [nn.Linear(13, 12), nn.ReLU(), nn.Linear(12, 8), nn.ReLU(),
     nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 3)]
]

learning_rate_arr = [.0001, .001, .01]
stopping_criteria_arr = ['trainLoss', 'valLoss', 'valAccuracy']


# # # Part A
single_run(hidden_layers[0], learning_rate_arr[2], stopping_criteria_arr[1], False)


# # # Part B

# Hidden Layers and Nodes: Average performance accross other parameters
# metrics_avg = np.zeros((3, 5))
# for a in range(len(hidden_layers)):
#     hidden_layer_a = hidden_layers[a]
#     metrics_matrix = np.zeros((9, 5))
#     count = 0
#     for b in range(len(stopping_criteria_arr)):
#         stopping_criteria_b = stopping_criteria_arr[b]
#         for c in range(len(learning_rate_arr)):
#             learning_rate_c = learning_rate_arr[c]
#             metrics_arr = single_run(
#                 hidden_layer_a, learning_rate_c, stopping_criteria_b, True)
#             metrics_matrix[count] = metrics_arr
#             count +=1


#     metrics_avg[a] = np.mean(metrics_matrix, axis=0)

# pd.DataFrame(np.around(metrics_avg.T, decimals=2)).to_csv("compare_hidden_layers_and_nodes.csv")


# Learning Rate: Average performance accross other parameters
# metrics_avg = np.zeros((3, 5))
# for a in range(len(learning_rate_arr)):
#     learning_rate_a = learning_rate_arr[a]
#     metrics_matrix = np.zeros((9, 5))
#     count = 0
#     for b in range(len(stopping_criteria_arr)):
#         stopping_criteria_b = stopping_criteria_arr[b]
#         for c in range(len(hidden_layers)):
#             hidden_layers_c = hidden_layers[c]
#             metrics_arr = single_run(
#                 hidden_layers_c, learning_rate_a, stopping_criteria_b, True)
#             metrics_matrix[count] = metrics_arr
#             count +=1


#     metrics_avg[a] = np.mean(metrics_matrix, axis=0)

# pd.DataFrame(np.around(metrics_avg.T, decimals=2)).to_csv("compare_learning_rate.csv")

# # Stopping Crteria: Average performance accross other parameters
# metrics_avg = np.zeros((3, 5))
# for a in range(len(stopping_criteria_arr)):
#     stopping_criteria_a = stopping_criteria_arr[a]
#     metrics_matrix = np.zeros((9, 5))
#     count = 0
#     for b in range(len(learning_rate_arr)):
#         learning_rate_b = learning_rate_arr[b]
#         for c in range(len(hidden_layers)):
#             hidden_layers_c = hidden_layers[c]
#             metrics_arr = single_run(
#                 hidden_layers_c, learning_rate_b, stopping_criteria_a, True)
#             metrics_matrix[count] = metrics_arr
#             count +=1


#     metrics_avg[a] = np.mean(metrics_matrix, axis=0)

# pd.DataFrame(np.around(metrics_avg.T, decimals=2)).to_csv("compare_stopping_criteria.csv")
