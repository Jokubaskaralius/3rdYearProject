import os
import numpy as np
import torch
from torch import optim
import torchvision.transforms as transforms
from classes import Dataset, logisticRegression
from utils import getImagePaths
from visualize import Visualize

# For reproducability
SEED = 42
if (type(SEED) == int):
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
def validate(labels_predicted, true_labels, arr, p_thresh=0.5):
    tp, tn, fp, fn = arr[0], arr[1], arr[2], arr[3]
    predicted_class = (labels_predicted >= p_thresh).long()
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
    lr = 0.005
    try:
        if (device.type == 'cuda'):
            loss_func = torch.nn.BCELoss().cuda()
            model = logisticRegression(n_input_features).cuda()
    except:
        loss_func = torch.nn.BCELoss(reduction="mean")
        model = logisticRegression(n_input_features)
    opt = optim.SGD(model.parameters(), lr=lr)
    return loss_func, model, opt


#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
def classifier():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Parameters
    params = {
        'batch_size': 10,
        'shuffle': True,
        'num_workers': 4,
        "pin_memory": True
    }
    max_epochs = 1

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
    visualize = Visualize()

    # Loop over epochs
    for epoch in range(max_epochs):
        #Training
        cost = train(training_generator, model, loss_func, opt, device)
        visualize.trainingLoss(epoch, cost)
        if (epoch + 1) % 1 == 0:
            print(f'epoch {epoch+1}, cost = {cost:.4f}')

        #Cross validation
        accuracy, cost = validation(validation_generator, model, loss_func,
                                    device)
        visualize.validationLoss(epoch, cost)
        if (epoch + 1) % 1 == 0:
            print(f'epoch {epoch+1}, accuracy = {accuracy*100:.4f}%')

    # Testing the final accuracy of the model
    data_ROC = list()
    accuracy, tp, tn, fp, fn = test(testing_generator,
                                    model,
                                    device,
                                    target_ROC=data_ROC)
    visualize.confusionMatrix(tp, tn, fp, fn, epoch + 1)
    visualize.ROC(data_ROC)
    if (epoch + 1) % 1 == 0:
        print(f'Final Classifier Accuracy = {accuracy*100:.4f}%')


def train(data_generator, model, loss_func, opt, device):
    epoch_loss = 0.0
    epoch_cost = 0.0
    # Training
    for local_batch, local_labels in data_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(
            device)

        local_labels = local_labels.view(-1, 1).float()

        # Model computations
        labels_predicted = model(local_batch).cuda()
        loss = loss_func(labels_predicted, local_labels).cuda()
        loss.backward()
        opt.step()
        opt.zero_grad()
        batch_loss = labels_predicted.shape[0] * loss.item()
        epoch_loss = epoch_loss + batch_loss

    epoch_cost = epoch_loss / len(data_generator.dataset)
    return epoch_cost


def validation(data_generator, model, loss_func, device):
    epoch_loss = 0.0
    epoch_cost = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in data_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(
                device), local_labels.to(device)

            local_labels = local_labels.view(-1, 1).float()

            # Model computations
            labels_predicted = model(local_batch).cuda()
            loss = loss_func(labels_predicted, local_labels).cuda()

            batch_loss = labels_predicted.shape[0] * loss.item()
            epoch_loss = epoch_loss + batch_loss

            tp, tn, fp, fn = validate(labels_predicted, local_labels,
                                      [tp, tn, fp, fn])

        epoch_cost = epoch_loss / len(data_generator.dataset)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy, epoch_cost


def test(data_generator, model, device, target_ROC=None):
    if target_ROC == None:
        probability_thresholds = torch.Tensor([0.5])
    else:
        probability_thresholds = torch.linspace(0, 1, steps=10)

    with torch.set_grad_enabled(False):
        best_accuracy = 0
        for p_thresh in probability_thresholds:
            tp, tn, fp, fn = 0, 0, 0, 0
            for local_batch, local_labels in data_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(
                    device), local_labels.to(device)

                local_labels = local_labels.view(-1, 1).float()

                # Model computations
                labels_predicted = model(local_batch).cuda()
                tp, tn, fp, fn = validate(labels_predicted,
                                          local_labels, [tp, tn, fp, fn],
                                          p_thresh=p_thresh)

            if target_ROC != None:
                ROC(tp, tn, fp, fn, p_thresh, target_ROC)

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            if (accuracy > best_accuracy):
                best_accuracy = accuracy
                threshold = p_thresh
                b_tp, b_tn, b_fp, b_fn = tp, tn, fp, fn
        print(
            f'Best Classifier Accuracy = {accuracy*100:.4f}%, with classification threshold = {threshold}'
        )
    return best_accuracy, b_tp, b_tn, b_fp, b_fn


def ROC(tp, tn, fp, fn, p_thresh, target_list):
    if (type(target_list) != list):
        print("Invalid target. Should be a list.")
        return -1
    true_positive_rate = tp / (tp + fn)
    false_positive_rate = fp / (fp + tn)
    result = [float(p_thresh), true_positive_rate, false_positive_rate]
    target_list.append(result)


classifier()
