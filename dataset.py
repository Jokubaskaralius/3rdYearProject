import nibabel as nib
import re
import torch
import numpy as np
from typing import List, Optional, Any, Tuple, Dict, Callable
from utils import PathManager


class DatasetManager:
    def __init__(self,
                 params: Dict[str, Any],
                 pathManager: PathManager,
                 transforms: Optional[List[List[Any]]] = None):
        if not isinstance(params, dict):
            raise TypeError("Expected dict; got %s" % type(params).__name__)
        if not params:
            raise ValueError("Expected %s dict; got empty dict" %
                             os.path.basename(__file__))

        self.transforms = transforms
        self.path_manager = pathManager
        self.image_paths = self.path_manager.unproc_image_paths()

    def process_images(self):
        for i, path in enumerate(self.image_paths):
            print("Path: ", path)
            sample = self._load_sample(path)
            print("Sample shape:", sample[0].shape)
            sample = self._apply_transforms(sample, self.transforms)
            processed_target_path = self.path_manager.append_proc_path(path)
            print("Sample shape pre-processed:", sample[0].shape)
            self._save_sample(sample, processed_target_path)
            print("processed: ", i + 1)

    def process_image(self, image_path: str):
        if not isinstance(image_path, str):
            raise TypeError("Expected str; got %s" % type(image_path).__name__)
        if not image_path:
            raise ValueError("Expected %s str; got empty str" %
                             os.path.basename(__file__))
        sample = self._load_sample(image_path)
        sample = self._apply_transforms(sample, self.transforms)
        processed_target_path = self.path_manager.append_proc_path(image_path)
        self._save_sample(sample, processed_target_path)

    @staticmethod
    def _load_sample(image_path: str) -> Tuple:
        if not isinstance(image_path, str):
            raise TypeError("Expected str; got %s" % type(image_path).__name__)
        if not image_path:
            raise ValueError("Expected %s str; got empty str" %
                             os.path.basename(__file__))
        img = nib.load(image_path)
        image_data = img.get_fdata(dtype=np.float32)
        img_header = img.header
        img_affine = img_header.get_best_affine()  #look into getting affine.
        return (image_data, img_affine, img_header)

    @staticmethod
    def _apply_transforms(sample: Tuple, transforms: List[List[Any]]) -> Tuple:
        if not isinstance(sample, tuple):
            raise TypeError("Expected tuple; got %s" % type(sample).__name__)
        image_data, img_affine, img_header = sample
        if (transforms is not None):
            for Transform_list in transforms:
                #item can either be a function or an argument list
                for item in Transform_list:
                    if (callable(item)):
                        Transform = item
                    else:
                        argx = item
                transform = Transform(sample, *argx)
                image_data = transform()
                sample = (image_data, img_affine, img_header)
        return (image_data, img_affine, img_header)

    @staticmethod
    def _save_sample(sample: Tuple, image_path: str):
        if not isinstance(sample, tuple):
            raise TypeError("Expected tuple; got %s" % type(sample).__name__)
        if not isinstance(image_path, str):
            raise TypeError("Expected str; got %s" % type(image_path).__name__)
        if not image_path:
            raise ValueError("Expected %s str; got empty str" %
                             os.path.basename(__file__))
        img = nib.Nifti1Image(sample[0], sample[1], sample[2])
        nib.save(img, image_path)

    def partition(self, shuffleSeed: Optional[int] = None) -> Dict[str, str]:
        partition = dict()
        processed_image_paths = self.path_manager.proc_image_paths()
        partition["dataset"] = processed_image_paths
        return partition

    def labels(self) -> Dict[str, str]:
        labels = dict()
        classes = ('grade1', 'grade2', 'grade3', 'grade4')
        for path in self.path_manager.proc_image_paths():
            data_class = re.findall("grade[1-9]", path)[0]
            if data_class == "grade1":
                labels[path] = torch.tensor([1, 0, 0, 0])
            elif data_class == "grade2":
                labels[path] = torch.tensor([0, 1, 0, 0])
            elif data_class == "grade3":
                labels[path] = torch.tensor([0, 0, 1, 0])
            elif data_class == "grade4":
                labels[path] = torch.tensor([0, 0, 0, 1])
        return labels


#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
#https://medium.com/@aakashns/image-classification-using-logistic-regression-in-pytorch-ebb96cc9eb79
#https://github.com/jcreinhold/niftidataset/blob/master/niftidataset/dataset.py
class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels):
        self.list_IDs = list_IDs
        if not isinstance(self.list_IDs, list):
            raise TypeError("Expected list; got %s" %
                            type(data_class_name).__name__)
        if not all(isinstance(id, str) for id in self.list_IDs):
            raise TypeError("Expected list of str; got non str list elements")
        if not self.list_IDs:
            raise ValueError("Expected list of str; got empty list")

        self.labels = labels
        if not isinstance(self.labels, dict):
            raise TypeError("Expected dict; got %s" % type(params).__name__)
        if not self.labels:
            raise ValueError("Expected %s dict; got empty dict" %
                             os.path.basename(__file__))

    #Denotes the total number of samples
    def __len__(self):
        return len(self.list_IDs)

    #Generates one sample of data
    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        if not isinstance(ID, str):
            raise TypeError("Expected str; got %s" % type(ID).__name__)

        img = nib.load(ID)
        image_data = img.get_fdata(dtype=np.float32)
        image_data = torch.from_numpy(image_data)
        image_data = torch.FloatTensor(image_data)

        X = image_data
        y = self.labels[ID]

        return X, y

    def _labels(self):
        labels = []
        temp = {}
        for item in self.labels:
            for idx, value in enumerate(self.labels[item]):
                if (value == 1):
                    labels.append(idx)
        return labels

    def _labels_one_hot(self):
        return self.labels