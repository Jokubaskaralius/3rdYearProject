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


def populate_matching_matrix(labels_predicted, true_labels, matching_matrix):
    predicted_class = torch.argmax(labels_predicted, dim=1)
    for idx, item in enumerate(predicted_class):
        matching_matrix[predicted_class[idx]][true_labels[
            idx]] = matching_matrix[predicted_class[idx]][true_labels[idx]] + 1
    return matching_matrix


# matrix - [[tp0, tn0, fp0, fn0], [tp1, tn1, fp1, fn1], ....]
def populate_confusion_matrix(matching_matrix, confusion_matrix):
    for idx_row, row in enumerate(matching_matrix):
        #get rid of the row and column of idx_row, to find true negative values
        tn_matrix = torch.cat(
            [matching_matrix[0:idx_row], matching_matrix[idx_row + 1:]])
        tn_matrix = torch.t(tn_matrix)
        tn_matrix = torch.cat([tn_matrix[0:idx_row], tn_matrix[idx_row + 1:]])
        tn_matrix = torch.t(tn_matrix)

        #true positives
        confusion_matrix[idx_row][0] = matching_matrix[idx_row][idx_row]
        #true negatives
        confusion_matrix[idx_row][1] = torch.sum(tn_matrix)
        #false positives
        confusion_matrix[idx_row][2] = torch.sum(
            matching_matrix[idx_row]) - matching_matrix[idx_row][idx_row]
        #false negatives
        col = torch.transpose(matching_matrix, 0, 1)[idx_row]
        confusion_matrix[idx_row][3] = torch.sum(col) - col[idx_row]

    return confusion_matrix


def derive_performance_measures(confusion_matrix, performance_matrix):
    for idx, item in enumerate(confusion_matrix):
        true_positive = confusion_matrix[idx][0]
        true_negative = confusion_matrix[idx][1]
        false_positive = confusion_matrix[idx][2]
        false_negative = confusion_matrix[idx][3]

        #Accuracy
        accuracy = torch.true_divide(
            true_positive + true_negative,
            true_positive + true_negative + false_positive + false_negative)
        performance_matrix[idx][0] = accuracy
        #Precision
        precision = torch.true_divide(true_positive,
                                      true_positive + false_positive)
        performance_matrix[idx][1] = precision
        #Recall
        recall = torch.true_divide(true_positive,
                                   true_positive + false_negative)
        performance_matrix[idx][2] = recall
        #F1-score
        f1_score = 2 * torch.true_divide(precision * recall,
                                         precision + recall)
        performance_matrix[idx][3] = f1_score
    return performance_matrix


#Use the predicted labels and true labels to check how many did the model get right
#labels_predicted - a tensor that contains the label predicted for each item in the batch
#true_labels - a tensor that stores the true class of the data item in the batch
#arr - true positive, true negative, false positive, false negative
#returns - true positive, true negative, false positive, false negative
# matrix - [[tp0, tn0, fp0, fn0], [tp1, tn1, fp1, fn1], ....]
# def derive_performance_measures(validation_matrix):
#     matrix = torch.zeros([4, 4], dtype=torch.int32)
#     performance_matrix = torch.zeros([4, 4], dtype=torch.int32)

#     for row in validation_matrix:
#         for col in row:

#     # #predicted_class = (labels_predicted >= p_thresh).long()
#     # predicted_class = torch.argmax(labels_predicted, dim=1)
#     # for idx, item in enumerate(predicted_class):
#     #     validation_matrix[predicted_class[idx]][
#     #         true_labels[idx]] = validation_matrix[predicted_class[idx]][
#     #             true_labels[idx]] + 1
#     #     # answer = predicted_class[idx].eq(true_labels[idx])
#     # return validation_matrix


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
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 4,
        "pin_memory": True
    }
    max_epochs = 1000

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
        matching_matrix, confusion_matrix, performance_matrix, cost = validation(
            validation_generator, model, loss_func, device, n_input_features)
        visualize.validationLoss(epoch, cost)
        # if (epoch + 1) % 1 == 0:
        #     print(f'epoch {epoch+1}, accuracy = {accuracy*100:.4f}%')

    # Testing the final accuracy of the model
    matching_matrix, confusion_matrix, performance_matrix = test(
        testing_generator, model, device, n_input_features, target_ROC=None)

    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.cpu().numpy()

    if isinstance(performance_matrix, torch.Tensor):
        performance_matrix = performance_matrix.cpu().numpy()

    visualize.confusionMatrix(matching_matrix, confusion_matrix,
                              performance_matrix, epoch + 1)

    # data_ROC = list()
    # visualize.ROC(data_ROC)
    # if (epoch + 1) % 1 == 0:
    #     print(f'Final Classifier Accuracy = {accuracy*100:.4f}%')
    print("Finished")


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
    matching_matrix = torch.zeros([4, 4], dtype=torch.int32)
    confusion_matrix = torch.zeros([4, 4], dtype=torch.int32)
    performance_matrix = torch.zeros([4, 4], dtype=torch.float32)
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

            matching_matrix = populate_matching_matrix(labels_predicted,
                                                       local_labels,
                                                       matching_matrix)

        confusion_matrix = populate_confusion_matrix(matching_matrix,
                                                     confusion_matrix)
        epoch_cost = epoch_loss / len(data_generator.dataset)

        performance_matrix = derive_performance_measures(
            confusion_matrix, performance_matrix)
    return matching_matrix, confusion_matrix, performance_matrix, epoch_cost


def test(data_generator, model, device, n_input_features, target_ROC=None):
    matching_matrix = torch.zeros([4, 4], dtype=torch.int32)
    confusion_matrix = torch.zeros([4, 4], dtype=torch.int32)
    performance_matrix = torch.zeros([4, 4], dtype=torch.float32)
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
                matching_matrix = populate_matching_matrix(
                    labels_predicted, local_labels, matching_matrix)

            if target_ROC != None:
                ROC(tp, tn, fp, fn, p_thresh, target_ROC)

            confusion_matrix = populate_confusion_matrix(
                matching_matrix, confusion_matrix)

            performance_matrix = derive_performance_measures(
                confusion_matrix, performance_matrix)

        #     accuracy = tp / (tp + tn)
        #     if (accuracy > best_accuracy):
        #         best_accuracy = accuracy
        #         threshold = p_thresh
        #         b_tp, b_tn, b_fp, b_fn = tp, tn, fp, fn
        # print(
        #     f'Best Classifier Accuracy = {accuracy*100:.4f}%, with classification threshold = {threshold}'
        # )
    return matching_matrix, confusion_matrix, performance_matrix


def ROC(tp, tn, fp, fn, p_thresh, target_list):
    if (type(target_list) != list):
        print("Invalid target. Should be a list.")
        return -1
    true_positive_rate = tp / (tp + fn)
    false_positive_rate = fp / (fp + tn)
    result = [float(p_thresh), true_positive_rate, false_positive_rate]
    target_list.append(result)


classifier(img_pre=False)
