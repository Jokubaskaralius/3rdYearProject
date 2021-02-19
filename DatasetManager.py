import os
import re
import random
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable, Any
import nibabel as nib
from torch.utils.data import Dataset

# PathManager
# Return all paths to images
# Images can be pre-processed and processed


class PathManager:
    def __init__(self):
        self.image_paths = self.unprocessed_image_paths()

    def __len__(self, ):
        return len(self.image_paths)

    def _split_filepath(self, filepath: str) -> Tuple[str, str, str]:
        path = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        base, ext = os.path.splitext(filename)
        if (ext == ".gz"):
            base, ext2 = os.path.splitext(base)
            ext = ext2 + ext
        return path, base, ext

    def _project_path(self) -> str:
        project_pathname = os.path.dirname(os.path.abspath(__file__))
        return project_pathname

    def _project_data_path(self) -> str:
        project_data_path = os.path.join(self._project_path(), "data")
        return project_data_path

    def _project_data_class_path(self) -> Tuple[str, str, str, str]:
        project_data_path = self._project_data_path()
        project_data_class_path_1 = os.path.join(project_data_path, "grade1")
        project_data_class_path_2 = os.path.join(project_data_path, "grade2")
        project_data_class_path_3 = os.path.join(project_data_path, "grade3")
        project_data_class_path_4 = os.path.join(project_data_path, "grade4")
        return (project_data_class_path_1, project_data_class_path_2,
                project_data_class_path_3, project_data_class_path_4)

    def _project_data_image_paths(self) -> List[str]:
        data_class_paths = self._project_data_class_path()
        image_paths = list()

        for data_class_path in data_class_paths:
            for idx, x in enumerate(os.walk(data_class_path)):
                if (idx == 0):
                    continue

                axial_MRI_data_path = re.findall("^.*T1-axial$", x[0])
                if (axial_MRI_data_path):
                    files = sorted([
                        f for f in os.listdir(axial_MRI_data_path[0])
                        if os.path.isfile(
                            os.path.join(axial_MRI_data_path[0], f))
                    ])
                    if (not files):
                        raise ValueError(
                            f'No files found in: {axial_MRI_data_path} \nSupported file image extension: .nii.gz'
                        )
                    for f in files:
                        image_paths.append(
                            os.path.join(axial_MRI_data_path[0], f))
        return image_paths

    def unprocessed_image_paths(self) -> List[str]:
        unprocessed_image_paths = []
        all_paths = self._project_data_image_paths()
        for path in all_paths:
            unprocessed_image_path = re.findall("_processed", path)
            if (not unprocessed_image_path):
                unprocessed_image_paths.append(path)
        return unprocessed_image_paths

    def _append_processed_image_path(self, filepath: str) -> str:
        path, base, ext = self._split_filepath(filepath)
        base = base + "_processed"
        filepath = os.path.join(path, base + ext)
        return filepath

    def processed_image_paths(self) -> List[str]:
        processed_image_paths = []
        all_paths = self._project_data_image_paths()
        for path in all_paths:
            processed_image_path = re.findall("_processed", path)
            if (processed_image_path):
                processed_image_paths.append(path)
        return processed_image_paths

    def paths_shuffle(self,
                      image_paths: List[str],
                      shuffleSeed: Optional[int] = None) -> List[str]:
        random.seed(shuffleSeed)
        random.shuffle(image_paths)
        return image_paths

    #100% to Training
    def create_partition(self,
                         shuffleSeed: Optional[int] = None) -> Dict[str, str]:
        partition = dict()
        processed_image_paths = self.processed_image_paths()
        processed_image_paths = self.paths_shuffle(processed_image_paths,
                                                   shuffleSeed=shuffleSeed)

        partition["dataset"] = processed_image_paths
        return partition

    #60 20 20 Training Validation Testing
    def create_partitions(self,
                          shuffleSeed: Optional[int] = None) -> Dict[str, str]:
        partition = dict()
        processed_image_paths = self.processed_image_paths()
        processed_image_paths = self.paths_shuffle(processed_image_paths,
                                                   shuffleSeed=shuffleSeed)

        training_dataset_count = round(len(processed_image_paths) * 0.6)
        training_dataset_paths = processed_image_paths[:training_dataset_count]
        for path in training_dataset_paths:
            processed_image_paths.remove(path)

        validation_dataset_count = round(len(processed_image_paths) * 0.5)
        validation_dataset_paths = processed_image_paths[:
                                                         validation_dataset_count]

        test_dataset_count = validation_dataset_count
        test_dataset_paths = processed_image_paths[test_dataset_count:]

        partition["train"] = training_dataset_paths
        partition["validation"] = validation_dataset_paths
        partition["test"] = test_dataset_paths
        return partition

    def create_labels(self) -> Dict[str, str]:
        labels = dict()
        classes = ('grade1', 'grade2', 'grade3', 'grade4')
        processed_image_paths = self.processed_image_paths()

        for path in processed_image_paths:
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


# DatasetManager
# It needs to be able to open an image and return data array
# Possible use:
############################################################
# manager = DatasetManager(
#    [  [Crop, []], [FeatureScaling, []], [SkullStrip, []],
#        [Resize, [(50, 50, 10)]], [ToTensor, []]])
#    ])
# manager.process_images()
############################################################


class DatasetManager(PathManager):
    def __init__(self, transforms: Optional[List[List[Any]]] = None):
        super().__init__()
        self.transforms = transforms
        if (not super().__len__()):
            raise ValueError(f'Number of source images to be non-zero')

    def process_images(self):
        count = 0
        for path in self.image_paths:
            sample = self._load_sample(path)
            sample = self._apply_transforms(sample)
            processed_target_path = self._append_processed_image_path(path)
            self._save_sample(processed_target_path, sample)
            count = count + 1
            print("Images processed:", count)
        print("Image processing finished")

    def process_image(self, image_path: str):
        sample = self._load_sample(image_path)
        sample = self._apply_transforms(sample)
        processed_target_path = self._append_processed_image_path(image_path)
        self._save_sample(processed_target_path, sample)
        print("Image processed")

    def _load_sample(self, image_path: str) -> Tuple:
        try:
            img = nib.load(image_path)
            img_header = img.header
            img_affine = img_header.get_best_affine()
        except:
            raise ValueError(
                f'Unexpected path: {image_path} \nExpected a path to NIFTI image. Supported image extension: .nii.gz'
            )
        image_data = img.get_fdata(dtype=np.float32)
        return (image_data, img_affine, img_header)

    def _apply_transforms(self, sample: Tuple):
        image_data, img_affine, img_header = sample
        if (self.transforms is not None):
            for Transform_list in self.transforms:
                #item can either be a function or an argument list
                for item in Transform_list:
                    if (callable(item)):
                        Transform = item
                    else:
                        argx = item
                transform = Transform(image_data, *argx)
                image_data = transform()
        return (image_data, img_affine, img_header)

    def _save_sample(self, image_path: str, sample: Tuple):
        try:
            img = nib.Nifti1Image(sample[0], sample[1], sample[2])
            nib.save(img, image_path)
        except:
            raise ValueError(
                f'Unexpected path: {image_path} \nExpected a path to NIFTI image. Supported image extension: .nii.gz'
            )
