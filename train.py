import os
import numpy as np
import torch
from torch import optim
import torchvision.transforms as transforms
from classes import Dataset, logisticRegression
from utils import getImagePaths
from visualize import Visualize
from sklearn.linear_model import LogisticRegression

# For reproducability
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# 60% training, 20% validation, 20% test
# Need to ensure that the training set has enough LGG data for classification?
# Also I shuffle the paths always. This might not be desirable,
# Because It is not possible to reproduce.
def createPartition():
    partition = dict()
    imagePaths = getImagePaths(MRIsequence="flair",
                               shuffle="yes",
                               shuffleSeed=SEED)

    trainingDatasetCount = round(len(imagePaths) * 0.6)
    trainingDatasetPaths = imagePaths[:trainingDatasetCount]
    for path in trainingDatasetPaths:
        imagePaths.remove(path)

    validationDatasetCount = round(len(imagePaths) * 0.5)
    validationDatasetPaths = imagePaths[:validationDatasetCount]

    testDatasetCount = validationDatasetCount
    testDatasetPaths = imagePaths[testDatasetCount:]

    partition["train"] = trainingDatasetPaths
    partition["validation"] = validationDatasetPaths
    partition["test"] = testDatasetPaths
    return partition


def createLabels():
    labels = dict()

    imagePaths = getImagePaths(MRIsequence="flair")
    if (imagePaths == -1):
        print("createLabels failed. imagePaths returned error status")
        return -1

    for path in imagePaths:
        dataClass = path.split(sep="/")[6]
        if dataClass == "HGG":
            labels[path] = 1.0
        elif dataClass == "LGG":
            labels[path] = 0.0
        else:
            print("createLabel. No such class exists. HGG or LGG")
            return -1

    return labels


#Use the predicted labels and true labels to check how many did the model get right
#labels_predicted - a tensor that contains the label predicted for each item in the batch
#true_labels - a tensor that stores the true class of the data item in the batch
#arr - true positive, true negative, false positive, false negative
#returns - true positive, true negative, false positive, false negative
def validate(labels_predicted, true_labels, arr):
    tp, tn, fp, fn = arr[0], arr[1], arr[2], arr[3]
    predicted_class = labels_predicted.round()
    arr = predicted_class.T.eq(true_labels)[0]
    for idx, item in enumerate(arr):
        label = int(true_labels[idx].item())
        item = bool(item.item())
        if (item is False and label == 0):
            fp = fp + 1
        elif (item is False and label == 1):
            fn = fn + 1
        elif (item is True and label == 1):
            tp = tp + 1
        else:
            tn = tn + 1
    return tp, tn, fp, fn


def get_model(n_input_features, device):
    lr = 0.001
    try:
        if (device.type == 'cuda'):
            loss_func = torch.nn.BCELoss().cuda()
            model = logisticRegression(n_input_features).cuda()
    except:
        loss_func = torch.nn.BCELoss()
        model = logisticRegression(n_input_features)
    opt = optim.SGD(model.parameters(), lr=lr)
    return loss_func, model, opt


#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
def train():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Parameters
    params = {'batch_size': 2, 'shuffle': True, 'num_workers': 6}
    max_epochs = 15

    # Datasets
    partition = createPartition()
    if (partition == -1):
        print("Creating partition failed")
        return -1
    labels = createLabels()
    if (labels == -1):
        print("Creating labels failed")
        return -1

    training_set = Dataset(partition['train'], labels)
    validation_set = Dataset(partition['validation'], labels)
    test_set = Dataset(partition['test'], labels)

    # Number of flattened features
    n_input_features = training_set[0][0].size()[0]

    # Generators
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    validation_generator = torch.utils.data.DataLoader(validation_set,
                                                       **params)
    testing_generator = torch.utils.data.DataLoader(test_set, **params)

    # Create a model
    loss_func, model, opt = get_model(n_input_features, device)

    #Visualization class
    visualize = Visualize(max_epochs)

    # Loop over epochs
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        epoch_cost = 0.0
        # Training
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(
                device), local_labels.to(device)

            local_labels = local_labels.view(-1, 1).float()

            # Model computations
            labels_predicted = model(local_batch)
            loss = loss_func(labels_predicted, local_labels)
            loss.backward()
            opt.step()
            opt.zero_grad()

            batch_loss = labels_predicted.shape[0] * loss.item()
            epoch_loss = epoch_loss + batch_loss

        epoch_cost = epoch_loss / len(training_set)
        visualize.trainingLoss(epoch, epoch_cost)
        if (epoch + 1) % 1 == 0:
            print(f'epoch {epoch+1}, cost = {epoch_cost:.4f}')

        # Validation
        with torch.set_grad_enabled(False):
            tp, tn, fp, fn = 0, 0, 0, 0
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(
                    device), local_labels.to(device)

                # Model computations
                labels_predicted = model(local_batch)
                tp, tn, fp, fn = validate(labels_predicted, local_labels,
                                          [tp, tn, fp, fn])

            visualize.confusionMatrix(tp, tn, fp, fn, epoch)
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            if (epoch + 1) % 1 == 0:
                print(f'epoch {epoch+1}, accuracy = {accuracy*100:.4f}%')

    # Testing the final accuracy of the model
    with torch.no_grad():
        tp, tn, fp, fn = 0, 0, 0, 0
        accuracy = 0
        for local_batch, local_labels in testing_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(
                device), local_labels.to(device)

            labels_predicted = model(local_batch)
            tp, tn, fp, fn = validate(labels_predicted, local_labels,
                                      [tp, tn, fp, fn])

        visualize.confusionMatrix(tp, tn, fp, fn, epoch)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        print(f'accuracy = {accuracy*100:.4f}%')


train()


def train2():
    X = list()
    Y = list()
    for local_batch, local_labels in training_generator:
        X.append(np.array(local_batch)[0])
        Y.append(np.array(local_labels)[0])
    X = np.array(X)
    Y = np.array(Y)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, Y)
    a = clf.score(X, Y)
    print(a)

    O = list()
    Z = list()
    for local_batch, local_labels in validation_generator:
        O.append(np.array(local_batch)[0])
        Z.append(np.array(local_labels)[0])
    O = np.array(O)
    Z = np.array(Z)
    print(clf.predict(O))
    print(Z)
    a = clf.score(O, Z)
    print(a)
    print(clf.coef_.shape)