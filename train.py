import os
import numpy as np
import torch
import re
from torch import optim
from classes import Dataset, logisticRegression
from visualize import Visualize
from DatasetManager import DatasetManager
from transforms import *
import matplotlib.pyplot as plt

# For reproducability
SEED = 42
if (type(SEED) == int):
    torch.manual_seed(SEED)
np.random.seed(SEED)

debug_image = 0


#Use the predicted labels and true labels to check how many did the model get right
#labels_predicted - a tensor that contains the label predicted for each item in the batch
#true_labels - a tensor that stores the true class of the data item in the batch
#arr - true positive, true negative, false positive, false negative
#returns - true positive, true negative, false positive, false negative
def validate(labels_predicted, true_labels, arr, p_thresh=0.5):
    tp, tn, fp, fn = arr[0], arr[1], arr[2], arr[3]
    #predicted_class = (labels_predicted >= p_thresh).long()
    predicted_class = torch.argmax(labels_predicted, dim=1)
    for idx, item in enumerate(predicted_class):
        answer = predicted_class[idx].eq(true_labels[idx])
        if (answer.item()):
            tp = tp + 1
        else:
            tn = tn + 1
    return tp, tn, fp, fn


def get_model(n_input_features, device):
    lr = 0.00005
    try:
        if (device.type == 'cuda'):
            loss_func = torch.nn.CrossEntropyLoss()
            model = logisticRegression(n_input_features).cuda()
    except:
        loss_func = torch.nn.CrossEntropyLoss()
        model = logisticRegression(n_input_features)
    opt = optim.Adam(model.parameters(), lr=lr)
    return loss_func, model, opt


#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
def classifier(img_pre=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Pre-process the images
    dataset_manager = DatasetManager([[Crop, []], [FeatureScaling, ["ZSN"]],
                                      [SkullStrip,
                                       []], [Resize, [(50, 50, 10)]],
                                      [ToTensor, []]])

    if (img_pre):
        dataset_manager.process_images()

    # Parameters
    params = {
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 4,
        "pin_memory": True
    }
    max_epochs = 2000

    # Pre-processed datasets
    partition = dataset_manager.create_partition(SEED)
    labels = dataset_manager.create_labels()

    training_set = Dataset(partition['train'], labels)
    validation_set = Dataset(partition['validation'], labels)
    test_set = Dataset(partition['test'], labels)

    if (debug_image):
        for image in training_set:
            plt.imshow(image[0].numpy())
            plt.show()

    # Number of flattened features
    n_input_features = torch.numel(training_set[0][0])

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
        cost = train(training_generator, model, loss_func, opt, device,
                     n_input_features)
        visualize.trainingLoss(epoch, cost)
        if (epoch + 1) % 1 == 0:
            print(f'epoch {epoch+1}, cost = {cost:.4f}')

        #Cross validation
        accuracy, cost = validation(validation_generator, model, loss_func,
                                    device, n_input_features)
        visualize.validationLoss(epoch, cost)
        if (epoch + 1) % 1 == 0:
            print(f'epoch {epoch+1}, accuracy = {accuracy*100:.4f}%')

    # Testing the final accuracy of the model
    data_ROC = list()
    accuracy, tp, tn, fp, fn = test(testing_generator,
                                    model,
                                    device,
                                    n_input_features,
                                    target_ROC=None)

    visualize.confusionMatrix(tp, tn, fp, fn, epoch + 1)
    visualize.ROC(data_ROC)
    if (epoch + 1) % 1 == 0:
        print(f'Final Classifier Accuracy = {accuracy*100:.4f}%')


def train(data_generator, model, loss_func, opt, device, n_input_features):
    epoch_loss = 0.0
    epoch_cost = 0.0
    # Training
    for local_batch, local_labels in data_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(
            device)

        local_batch = torch.reshape(local_batch,
                                    (len(local_batch), n_input_features))
        local_labels = torch.max(local_labels, 1)[1]

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


def validation(data_generator, model, loss_func, device, n_input_features):
    epoch_loss = 0.0
    epoch_cost = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in data_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(
                device), local_labels.to(device)

            local_batch = torch.reshape(local_batch,
                                        (len(local_batch), n_input_features))
            local_labels = torch.max(local_labels, 1)[1]

            # Model computations
            labels_predicted = model(local_batch).cuda()
            loss = loss_func(labels_predicted, local_labels).cuda()

            batch_loss = labels_predicted.shape[0] * loss.item()
            epoch_loss = epoch_loss + batch_loss

            tp, tn, fp, fn = validate(labels_predicted, local_labels,
                                      [tp, tn, fp, fn])

        epoch_cost = epoch_loss / len(data_generator.dataset)

        accuracy = tp / (tp + tn)
    return accuracy, epoch_cost


def test(data_generator, model, device, n_input_features, target_ROC=None):
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

                local_batch = torch.reshape(
                    local_batch, (len(local_batch), n_input_features))
                local_labels = torch.max(local_labels, 1)[1]

                # Model computations
                labels_predicted = model(local_batch).cuda()
                tp, tn, fp, fn = validate(labels_predicted,
                                          local_labels, [tp, tn, fp, fn],
                                          p_thresh=p_thresh)

            if target_ROC != None:
                ROC(tp, tn, fp, fn, p_thresh, target_ROC)

            accuracy = tp / (tp + tn)
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


classifier(img_pre=False)
