import argparse
import random
import json
import numpy as np
import torch
from typing import Dict, Any
from torch import optim
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold, KFold

from utils import *
from dataset import DatasetManager, Dataset
from models import CNNModel, Selector
from transforms import TransformManager
from visualize import Visualize


class Trainer:
    def __init__(self, dataset, config, hyper_param, export_path: str):
        self.dataset = dataset
        self._load_config(config)
        self._load_param(hyper_param)
        self._device()
        self._model(hyper_param)
        self.kfold = StratifiedKFold(n_splits=self.k_folds,
                                     shuffle=self.shuffle)
        self.training_cost = 0.0
        self.validation_cost = 0.0

        self.export_path = export_path

    def train(self):
        verbosePrint(f'Training start')
        verbosePrint('--------------------------------')
        for epoch in range(self.epoch_num):
            fold_total_training_cost = 0.0
            fold_total_validation_cost = 0.0
            for fold, (train_ids, validation_ids) in enumerate(
                    self.kfold.split(torch.zeros(self.dataset.__len__()),
                                     self.dataset._labels())):
                verbosePrint(f'FOLD {fold}')
                verbosePrint('--------------------------------')

                train_subsampler = torch.utils.data.SubsetRandomSampler(
                    train_ids)
                validation_subsampler = torch.utils.data.SubsetRandomSampler(
                    validation_ids)

                train_loader = torch.utils.data.DataLoader(
                    self.dataset,
                    **self._training_kwargs(),
                    sampler=train_subsampler)
                validation_loader = torch.utils.data.DataLoader(
                    self.dataset,
                    **self._validation_kwargs(),
                    sampler=validation_subsampler)

                fold_training_cost = self.train_step(train_loader)
                fold_validation_cost = self.validation_step(validation_loader)

                verbosePrint("training cost: ", fold_training_cost)
                verbosePrint("validation cost: ", fold_validation_cost)

                fold_total_training_cost = fold_total_training_cost + fold_training_cost
                fold_total_validation_cost = fold_total_validation_cost + fold_validation_cost

            verbosePrint("<><><><><><><><><><><><><><><><><><><>")
            verbosePrint(f'\nTOTAL')
            # Compute final Training and Validation error per all folds for single epoch
            self.training_cost = fold_total_training_cost / self.k_folds
            self.validation_cost = fold_total_validation_cost / self.k_folds
            self.epoch = epoch

            verbosePrint("training cost: ", self.training_cost)
            verbosePrint("validation cost: ", self.validation_cost)

            # Export Learning Curve data
            self._export_data(self._training_error(), self.export_path,
                              self.export_t_e_filename)
            self._export_data(self._validation_error(), self.export_path,
                              self.export_v_e_filename)
            verbosePrint("<><><><><><><><><><><><><><><><><><><>")

        verbosePrint(f'Training end')
        verbosePrint('--------------------------------')

        self._save_model()
        verbosePrint(f'Model: "{self.saved_model_filename}" saved.')
        verbosePrint('--------------------------------')

    def train_step(self, data_loader):
        fold_loss = 0.0
        for local_batch, local_labels in data_loader:
            local_batch, local_labels = local_batch.to(
                self.device), local_labels.to(
                    self.device)  # Transfer batch to GPU if GPU available

            #local_batch = torch.reshape(local_batch, (len(local_batch), input_dim))
            local_batch = local_batch.view(len(local_batch), 1, 100, 100, 56)
            local_labels = torch.max(local_labels, 1)[1]

            # Model computations
            labels_predicted = self.model(local_batch).to(self.device)
            loss = self.loss_func(labels_predicted,
                                  local_labels).to(self.device)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Cost per Epoch
            batch_loss = labels_predicted.shape[0] * loss.item()
            fold_loss = fold_loss + batch_loss

        fold_cost = fold_loss / len(data_loader)
        return fold_cost

    def validation_step(self, data_loader):
        fold_loss = 0.0
        matching_matrix = torch.zeros([4, 4], dtype=torch.int32)
        confusion_matrix = torch.zeros([4, 4], dtype=torch.int32)
        performance_matrix = torch.zeros([4, 4], dtype=torch.float32)

        with torch.set_grad_enabled(False):
            for local_batch, local_labels in data_loader:
                local_batch, local_labels = local_batch.to(
                    self.device), local_labels.to(
                        self.device)  # Transfer to GPU

                local_batch = local_batch.view(len(local_batch), 1, 100, 100,
                                               56)
                local_labels = torch.max(local_labels, 1)[1]

                # Model computations
                labels_predicted = self.model(local_batch).to(self.device)
                loss = self.loss_func(labels_predicted,
                                      local_labels).to(self.device)
                # Cost per Epoch
                batch_loss = labels_predicted.shape[0] * loss.item()
                fold_loss = fold_loss + batch_loss

            fold_cost = fold_loss / len(data_loader)
        return fold_cost

    def _load_config(self, config):
        self.use_GPU = to_bool(config["use_GPU"])
        self.save_model = to_bool(config["save_model"])
        self.saved_model_filename = config["saved_model_filename"]
        self.export_t_e_filename = config["training_error_filename"]
        self.export_v_e_filename = config["validation_error_filename"]

    def _load_param(self, hyper_param):
        self.batch_size = hyper_param["batch_size"]
        self.num_workers = hyper_param["num_workers"]
        self.shuffle = to_bool(hyper_param["shuffle"])

        self.epoch_num = hyper_param["epoch_num"]
        self.k_folds = hyper_param["k_folds"]

        self.model = hyper_param["model"]
        self.optimizer = hyper_param["optimizer"]
        self.lr = hyper_param["learning_rate"]
        self.loss_func = hyper_param["loss_func"]

    def _device(self) -> bool:
        use_cuda = torch.cuda.is_available() and self.use_GPU
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        return use_cuda

    def _model(self, hyper_param):
        model_selector = Selector(hyper_param)
        self.model, self.optimizer, self.loss_func = model_selector()
        self.model.to(self.device)
        return

    def _training_kwargs(self):
        training_kwargs = {"batch_size": self.batch_size}
        if (self._device()):
            cuda_kwargs = {'num_workers': self.num_workers, "pin_memory": True}
            training_kwargs.update(cuda_kwargs)
        return training_kwargs

    def _validation_kwargs(self):
        validation_kwargs = {"batch_size": self.batch_size}
        if (self._device()):
            cuda_kwargs = {'num_workers': self.num_workers, "pin_memory": True}
            validation_kwargs.update(cuda_kwargs)
        return validation_kwargs

    def _training_error(self):
        export_data = {"x": self.epoch, "y": self.training_cost}
        return export_data

    def _validation_error(self):
        export_data = {"x": self.epoch, "y": self.validation_cost}
        return export_data

    @staticmethod
    def _export_data(data: Dict, path: str, filename: str):
        if not isinstance(data, dict):
            raise TypeError("Expected dict; got %s" % type(params).__name__)
        if not data:
            raise ValueError("Expected dict; got empty dict")

        path = os.path.join(path, filename + ".json")
        try:
            data_json = load_JSON(path)
            data_json.append(data)
            export_JSON(data_json, path)
        except:
            export_JSON([data], path)
        return

    def _save_model(self):
        if (self.save_model):
            torch.save(self.model.state_dict(), self.saved_model_filename)


def main(config: Dict[str, Any], hyper_param: Dict[str, Any]):
    # Config
    seed = config["seed"]

    # Initial seed
    # 1) Pytorch dataLoader
    # 2) Numpy random
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate project paths
    path_manager = PathManager(config["pathManager"])
    export_path = path_manager.visuals_data_dir()

    # Generate transforms
    transforms_manager = TransformManager(config["transformManager"])
    transforms = transforms_manager.transforms()

    # Pre-process dataset
    dataset_manager = DatasetManager(config["datasetManager"], path_manager,
                                     transforms)
    #dataset_manager.process_images()

    # Form training/validation partitions
    partition = dataset_manager.partition(seed)
    labels = dataset_manager.labels()
    dataset = Dataset(partition['dataset'], labels)

    trainer = Trainer(dataset, config["trainer"], hyper_param, export_path)
    trainer.train()


# End of Main

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

    verbose = config["verbose"]
    # Verbose mode
    if verbose:

        def verbosePrint(*args):
            for arg in args:
                print(arg, )
    else:
        verbosePrint = lambda *a: Nones

    # Delete the visualization things..

    main(config, hyper_param)
    # End of command line arguments
