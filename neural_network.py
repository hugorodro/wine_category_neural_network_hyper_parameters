from os import error
import torch
from torch import nn

torch.manual_seed(1)


class NeuralNetwork(nn.Module):
    def __init__(self, stack):
        super(NeuralNetwork, self).__init__()
        self.hiddenLayers = nn.Linear(13, 12)

        self.layers = nn.Sequential(*stack)

    def forward(self, x):
        predictions = self.layers(x)

        return predictions


# Train
def train_epoch(dl, model, loss_fn, optimizer):

    for batch, (X, y) in enumerate(dl):

        # Compute prediction error
        pred = model(X.float())
        # print(pred)
        # print(y)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss.item()
    # print(final_loss)


def test(dl, model, loss_fn, val):
    size = len(dl.dataset)
    num_batches = len(dl)
    model.eval()
    test_loss, correct = 0, 0
    predictions = []
    true = []
    with torch.no_grad():
        for X, y in dl:
            pred = model(X.float())
            predictions = pred.argmax(1).numpy
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            predictions = pred.argmax(1).numpy()
            true = y.numpy()

    test_loss /= num_batches
    correct /= size

    # print(
    #     f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct, predictions, true


def train(dlTrain, dlTest, model, loss_fn, optimizer, epochs, stoppingCriteria, limit):
    training_loss = []
    validation_loss = []
    accuracy = []

    prevTrainingLoss = 10000000
    prevValLoss = 1000000
    prevAccuracy = 0

    countWorse = 0

    predictions, true = None, None

    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")

        thisTrainingLoss = train_epoch(dlTrain, model, loss_fn, optimizer)
        thisValLoss, thisAccuracy, predictions, true = test(
            dlTest, model, loss_fn, True)

        training_loss.append(thisTrainingLoss)
        validation_loss.append(thisValLoss)
        accuracy.append(thisAccuracy)

        numEpochs = 0

        if t > 10:
            if stoppingCriteria == 'trainLoss':
                if thisTrainingLoss > prevTrainingLoss:
                    countWorse += 1
            if stoppingCriteria == 'valLoss':
                if thisValLoss > prevValLoss:
                    countWorse += 1
            if stoppingCriteria == 'valAccuracy':
                if thisAccuracy <= prevAccuracy:
                    countWorse += 1

        numEpochs = t+1
        if limit != None:
            if countWorse > limit:
                break
        prevTrainingLoss, prevValLoss, prevAccuracy = thisTrainingLoss, thisValLoss, thisAccuracy

    return numEpochs, training_loss, validation_loss, accuracy, predictions, true
