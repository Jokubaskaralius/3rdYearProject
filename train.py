import argparse
import random
import json
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold, KFold

from utils import to_bool
from dataset import DatasetManager, Dataset
from models import CNNModel
from transforms import *
from visualize import Visualize


class Trainer:
    def __init__(self):
        self.device = device
        self.model = Model
        self.optimizer = optim
        self.data_loader = data_loader
        self.loss_func = loss_func

    def train(self):
        epoch_loss = 0.0
        loss_func = torch.nn.CrossEntropyLoss()
        # Training
        for local_batch, local_labels in data_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(
                device), local_labels.to(device)

            #local_batch = torch.reshape(local_batch, (len(local_batch), input_dim))
            local_batch = local_batch.view(len(local_batch), 1, 100, 100, 56)
            local_labels = torch.max(local_labels, 1)[1]

            # Model computations
            labels_predicted = model(local_batch)
            loss = loss_func(labels_predicted, local_labels).to(device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Cost per Epoch
            batch_loss = labels_predicted.shape[0] * loss.item()
            epoch_loss = epoch_loss + batch_loss

        epoch_cost = epoch_loss / len(data_loader)
        return epoch_cost

    def validate(self):
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

                # local_batch = torch.reshape(local_batch,
                #                             (len(local_batch), input_dim))
                local_batch = local_batch.view(len(local_batch), 1, 100, 100,
                                               56)
                local_labels = torch.max(local_labels, 1)[1]

                # Model computations
                labels_predicted = model(local_batch).to(device)
                loss = loss_func(labels_predicted, local_labels).to(device)

                matching_matrix = populate_matching_matrix(
                    labels_predicted, local_labels, matching_matrix)

                # Cost per Epoch
                batch_loss = labels_predicted.shape[0] * loss.item()
                epoch_loss = epoch_loss + batch_loss

            confusion_matrix = populate_confusion_matrix(
                matching_matrix, confusion_matrix)

            performance_matrix = derive_performance_measures(
                confusion_matrix, performance_matrix)

            epoch_cost = epoch_loss / len(data_loader)
        return matching_matrix, confusion_matrix, performance_matrix, epoch_cost


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
    loss_func = torch.nn.CrossEntropyLoss()

    # Training
    for local_batch, local_labels in data_loader:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(
            device)

        #local_batch = torch.reshape(local_batch, (len(local_batch), input_dim))
        local_batch = local_batch.view(len(local_batch), 1, 100, 100, 56)
        local_labels = torch.max(local_labels, 1)[1]

        # Model computations
        labels_predicted = model(local_batch)
        loss = loss_func(labels_predicted, local_labels).to(device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Cost per Epoch
        batch_loss = labels_predicted.shape[0] * loss.item()
        epoch_loss = epoch_loss + batch_loss

    epoch_cost = epoch_loss / len(data_loader)
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

            # local_batch = torch.reshape(local_batch,
            #                             (len(local_batch), input_dim))
            local_batch = local_batch.view(len(local_batch), 1, 100, 100, 56)
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

        epoch_cost = epoch_loss / len(data_loader)
    return matching_matrix, confusion_matrix, performance_matrix, epoch_cost


#https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
#https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
#https://github.com/alejandrodebus/Pytorch-Utils/blob/master/cross_validation.py
#https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation
def main(config, hyper_param):
    # CONFIG
    verbose = to_bool(config["verbose"])
    use_GPU = to_bool(config["use_GPU"])
    seed = config["seed"]
    save_model = to_bool(config["save_model"])
    saved_model_filename = config["saved_model_filename"]
    # End of CONFIG

    # HYPER_PARAMS
    batch_size = hyper_param["batch_size"]
    num_workers = hyper_param["num_workers"]
    shuffle = hyper_param["shuffle"]
    k_folds = hyper_param["k_folds"]
    model = hyper_param["model"]
    optimizer = hyper_param["optimizer"]
    lr = hyper_param["learning_rate"]
    epoch_num = hyper_param["epoch_num"]
    loss_func = hyper_param["loss_func"]
    # End of HYPER_PARAMS

    # VERBOSE
    if verbose:

        def verbosePrint(*args):
            for arg in args:
                print(arg, )
    else:
        verbosePrint = lambda *a: None
    # End of VERBOSE

    # SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    # End of SEED

    # GPU / CPU device setup
    use_cuda = torch.cuda.is_available() and use_GPU
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    # End of GPU

    # Training/Validation args
    training_kwargs = {'batch_size': batch_size}
    validation_kwargs = training_kwargs
    if (use_cuda):
        cuda_kwargs = {
            'num_workers': num_workers,
            'shuffle': shuffle,
            "pin_memory": True
        }
        training_kwargs.update(cuda_kwargs)
        validation_kwargs.update(cuda_kwargs)
    #

    # Dataset
    dataset_manager = DatasetManager([
        [FeatureScaling, ["MM"]],
        [Crop, []],
        [Resize, [(100, 100, 56)]],
        [SkullStrip, []],
        #[ToTensor, []]
    ])
    #dataset_manager.process_images()

    partition = dataset_manager.partition(seed)
    labels = dataset_manager.create_labels()
    dataset = Dataset(partition['dataset'], labels)
    # End of Dataset

    y = []
    for item in dataset:
        for idx, value in enumerate(item[1]):
            if (value == 1):
                y.append(idx)

    # K-folds cross validation
    kfold = StratifiedKFold(n_splits=k_folds,
                            shuffle=shuffle,
                            random_state=seed)
    # End of K-folds

    # Model
    input_dim = torch.numel(dataset[0][0])
    output_dim = 4
    #model = logisticRegression(input_dim, output_dim).to(device)
    model = CNNModel(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # End of Model

    visualize = Visualize()

    for epoch in range(args.epoch):
        training_cost = 0
        validation_cost = 0
        performance_measure_list = list()

        verbosePrint(f'EPOCH {epoch+1}')
        verbosePrint("<><><><><><><><><><><><><><><><><><><>")
        for fold, (train_ids, validation_ids) in enumerate(
                kfold.split(torch.zeros(len(dataset)), y)):
            verbosePrint(f'FOLD {fold}')
            verbosePrint('--------------------------------')
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            validation_subsampler = torch.utils.data.SubsetRandomSampler(
                validation_ids)

            # Define data loaders for training and testing data in this fold
            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, sampler=train_subsampler)
            validation_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=validation_subsampler)

            fold_training_cost = training(device, model, optimizer,
                                          train_loader, input_dim)
            matching_matrix, confusion_matrix, performance_matrix, fold_validation_cost = validation(
                device, validation_loader, model, input_dim)

            verbosePrint("training cost: ", fold_training_cost)
            verbosePrint("validation cost: ", fold_validation_cost)

            training_cost = training_cost + fold_training_cost
            validation_cost = validation_cost + fold_validation_cost

            performance_measure_list.append(
                [matching_matrix, confusion_matrix, performance_matrix])

        # The average training and validation costs for all folds
        verbosePrint(f'\nTOTAL')
        training_cost = training_cost / k_folds
        validation_cost = validation_cost / k_folds
        verbosePrint("training cost: ", training_cost)
        verbosePrint("validation cost: ", validation_cost)
        verbosePrint("<><><><><><><><><><><><><><><><><><><>")
        visualize.trainingLoss(epoch, training_cost)
        visualize.validationLoss(epoch, validation_cost)
        visualize.confusionMatrix(performance_measure_list, epoch)
        performance_measure_list.clear()

# Save Model
    if (args.save_model):
        torch.save(model.state_dict(), "./TrainedTest.pt")


# End of Save Model

if __name__ == "__main__":
    #Classifier training command line arguments

    parser = argparse.ArgumentParser(
        description="Train the Brain Tumor Classifier by WHO Grade")

    parser.add_argument(
        "--config",
        action="store",
        default="./config.json",
        dest="config_path",
        help=
        "Project configuration file, to configure project software components")
    parser.add_argument("--hyper_params",
                        action="store",
                        default="./hyper_param.json",
                        dest="hyper_param_path",
                        help="Disables training on GPU")
    args = parser.parse_args()

    config = json.load(open(args.config_path))
    hyper_param = json.load(open(args.hyper_param_path))

    if not isinstance(config, dict):
        raise TypeError("Expected dict; got %s" % type(params).__name__)
    if not config:
        raise ValueError("Expected %s dict; got empty dict" %
                         os.path.basename(__file__))
    if not isinstance(hyper_param, dict):
        raise TypeError("Expected dict; got %s" % type(params).__name__)
    if not hyper_param:
        raise ValueError("Expected %s dict; got empty dict" %
                         os.path.basename(__file__))

    main(config["trainer"], hyper_param)
    # End of command line arguments
