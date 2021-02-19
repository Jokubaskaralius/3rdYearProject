import argparse
import random
import numpy as np
import torch
from torch import optim
from sklearn.model_selection import KFold

from DatasetManager import DatasetManager
from classes import Dataset, logisticRegression
from transforms import *
from visualize import Visualize


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


def training(device, model, optimizer, data_loader, input_dim):
    epoch_loss = 0.0
    test = 0
    loss_func = torch.nn.CrossEntropyLoss()
    # Training
    for local_batch, local_labels in data_loader:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(
            device)

        local_batch = torch.reshape(local_batch, (len(local_batch), input_dim))
        local_labels = torch.max(local_labels, 1)[1]

        # Model computations
        labels_predicted = model(local_batch).to(device)
        loss = loss_func(labels_predicted, local_labels).to(device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Cost per Epoch
        batch_loss = labels_predicted.shape[0] * loss.item()
        epoch_loss = epoch_loss + batch_loss

    epoch_cost = epoch_loss / len(data_loader.dataset)
    return epoch_cost


#https://medium.com/apprentice-journal/evaluating-multi-class-classifiers-12b2946e755b
#https://www.youtube.com/watch?v=gJo0uNL-5Qw&t=343s
def validation(device, data_loader, model, input_dim):
    epoch_loss = 0.0
    matching_matrix = torch.zeros([4, 4], dtype=torch.int32)
    confusion_matrix = torch.zeros([4, 4], dtype=torch.int32)
    performance_matrix = torch.zeros([4, 4], dtype=torch.float32)

    loss_func = torch.nn.CrossEntropyLoss()
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in data_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(
                device), local_labels.to(device)

            local_batch = torch.reshape(local_batch,
                                        (len(local_batch), input_dim))
            local_labels = torch.max(local_labels, 1)[1]

            # Model computations
            labels_predicted = model(local_batch).to(device)
            loss = loss_func(labels_predicted, local_labels).to(device)

            matching_matrix = populate_matching_matrix(labels_predicted,
                                                       local_labels,
                                                       matching_matrix)

            # Cost per Epoch
            batch_loss = labels_predicted.shape[0] * loss.item()
            epoch_loss = epoch_loss + batch_loss

        confusion_matrix = populate_confusion_matrix(matching_matrix,
                                                     confusion_matrix)

        performance_matrix = derive_performance_measures(
            confusion_matrix, performance_matrix)

        epoch_cost = epoch_loss / len(data_loader.dataset)

    return matching_matrix, confusion_matrix, performance_matrix, epoch_cost


#https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
#https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
#https://github.com/alejandrodebus/Pytorch-Utils/blob/master/cross_validation.py
#https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation
def main():
    #Classifier training command line arguments
    parser = argparse.ArgumentParser(
        description="Train the Brain Tumor Classifier by WHO Grade")
    parser.add_argument("--no-GPU",
                        action="store_true",
                        default=False,
                        help="Disables training on GPU")
    parser.add_argument("--dry-run",
                        action="store_true",
                        default=False,
                        help="Test a single training iteration")
    parser.add_argument("--batch-size",
                        type=int,
                        default=1,
                        metavar='N',
                        help="Input batch size for training (default: 1)")
    parser.add_argument("--epoch",
                        type=int,
                        default=100,
                        metavar='N',
                        help="Number of epochs for training (default: 100)")
    parser.add_argument("--lr",
                        type=float,
                        default=0.00005,
                        metavar='F',
                        help="Optimization Learning Rate (default: 0.00005)")
    parser.add_argument(
        "--seed",
        type=int,
        metavar='N',
        help=
        "Seed used for traning/shuffling (default: Null - generate a random seed)"
    )
    parser.add_argument(
        '--log-state',
        type=int,
        default=10,
        metavar='N',
        help='Log training state after N iterations (default: 10)')
    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='Save the model & hyperparameters')
    args = parser.parse_args()
    # End of command line arguments

    # SEED
    if (not args.seed):
        args.seed = int(random.random() * 10000)
    # Set the seed of torch and numpy
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # End of SEED

    # GPU / CPU device setup
    use_cuda = torch.cuda.is_available() and not args.no_GPU
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    # End of GPU

    # Training/Validation args
    training_kwargs = {'batch_size': args.batch_size}
    validation_kwargs = training_kwargs
    if (use_cuda):
        cuda_kwargs = {'num_workers': 4, 'shuffle': True, "pin_memory": True}
        training_kwargs.update(cuda_kwargs)
        validation_kwargs.update(cuda_kwargs)
    #

    # Dataset
    dataset_manager = DatasetManager([
        #[Crop, []],
        [FeatureScaling, ["MM"]],
        #[SkullStrip, []],
        [Resize, [(50, 50, 10)]],
        [ToTensor, []]
    ])
    dataset_manager.process_images()
    partition = dataset_manager.create_partition(args.seed)
    labels = dataset_manager.create_labels()

    dataset = Dataset(partition['dataset'], labels)
    # End of Dataset

    # K-folds cross validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    # End of K-folds

    # Model
    input_dim = torch.numel(dataset[0][0])
    output_dim = 4
    model = logisticRegression(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # End of Model

    visualize = Visualize()

    for epoch in range(args.epoch):
        training_cost = 0
        validation_cost = 0
        performance_measure_list = list()
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

            print(f'FOLD {fold}')
            print('--------------------------------')
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in this fold
            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=10, sampler=train_subsampler)
            test_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=10,
                                                      sampler=test_subsampler)

            print(f'Starting epoch {epoch+1}')

            fold_training_cost = training(device, model, optimizer,
                                          train_loader, input_dim)
            matching_matrix, confusion_matrix, performance_matrix, fold_validation_cost = validation(
                device, test_loader, model, input_dim)

            training_cost = training_cost + fold_training_cost
            validation_cost = validation_cost + fold_validation_cost

            performance_measure_list.append(
                [matching_matrix, confusion_matrix, performance_matrix])

        # The average training and validation costs for all folds
        training_cost = training_cost / k_folds
        validation_cost = validation_cost / k_folds

        visualize.trainingLoss(epoch, training_cost)
        visualize.validationLoss(epoch, validation_cost)
        visualize.confusionMatrix(performance_measure_list, epoch)
        performance_measure_list.clear()

# Save Model
    if (args.save_model):
        torch.save(model.state_dict(), "./TrainedTest.pt")


# End of Save Model

if __name__ == "__main__":
    main()