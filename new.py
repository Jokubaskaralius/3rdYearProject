import os
import re
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable, Any
import nibabel as nib
from torch.utils.data import Dataset
from transforms import Transforms, Resize, SkullStrip

# PathManager
# Return all paths to images
# Images can be pre-processed and processed


class PathManager:
    def __init__(self):
        self.image_paths = self._project_data_image_paths(shuffle=False,
                                                          shuffleSeed=None)

    def __repr__(self):
        pass

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

    def _project_data_image_paths(
            #This shouldn't be shuffling paths. The ImagePreprocessor should
            #Only a temporary meassure.
            self,
            shuffle: Optional[bool] = False,
            shuffleSeed: Optional[int] = None) -> List[str]:
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
                    image_paths.append(
                        os.path.join(axial_MRI_data_path[0], files[0]))

        if (shuffle):
            random.seed(shuffleSeed)
            random.shuffle(image_paths)
        return image_paths

    def _project_processed_image_path(self, filepath: str) -> str:
        path, base, ext = self._split_filepath(filepath)
        base = base + "_processed"
        filepath = os.path.join(path, base + ext)
        return filepath

    #60 20 20 Training Validation Testing
    def create_partition(self) -> Dict[str, str]:
        partition = dict()
        image_paths = self.image_paths

        training_dataset_count = round(self.__len__() * 0.6)
        training_dataset_paths = image_paths[:training_dataset_count]
        for path in training_dataset_paths:
            image_paths.remove(path)

        validation_dataset_count = round(self.__len__() * 0.5)
        validation_dataset_paths = image_paths[:validation_dataset_count]

        test_dataset_count = validation_dataset_count
        test_dataset_paths = image_paths[test_dataset_count:]

        partition["train"] = training_dataset_paths
        partition["validation"] = validation_dataset_paths
        partition["test"] = test_dataset_paths
        return partition

    def createLabels(self) -> Dict[str, str]:
        labels = dict()
        classes = ('grade1', 'grade2', 'grade3', 'grade4')

        for path in self.image_paths:
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
#


class DatasetManager(PathManager):
    def __init__(self,
                 transforms: Optional[List[List[Any]]] = None):  #Callable,
        #Optional[List[Any]]]]] = None):
        super().__init__()
        self.transforms = transforms
        if (not super().__len__()):
            raise ValueError(f'Number of source images to be non-zero')

    def process_images(self):
        for path in self.image_paths:
            sample = self._load_sample(path)
            sample = self._apply_transform(sample)
            processed_target_path = self._project_processed_image_path(path)
            #self._save_sample(processed_target_path, sample)

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

    def _apply_transform(self, sample: Tuple):
        image_data, img_affine, img_header = sample
        if (self.transforms is not None):
            for Transform_list in self.transforms:
                #item can either be a function or an argument list
                for item in Transform_list:
                    if (callable(item)):
                        Transform = item
                    else:
                        argx = item
                transform = Transform(
                    image_data, *argx)  #transform(self, image_data, *argx)
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


manager = DatasetManager([
    #[Transforms.crop, []],
    #[Transforms.mean_normalize, []],
    [SkullStrip, []],
    #[Resize, [(50, 50, 10)]],
    #[Transforms.to_tensor, []]
])
manager.process_images()
# test_path = "/home/jokubas/DevWork/3rdYearProject/data/grade1/00002/T1-axial/_5_3d_spgr_volume.nii.gz"
# a = manager.load_sample(test_path)
# manager.save_sample(
#     "/home/jokubas/DevWork/3rdYearProject/data/grade1/00002/T1-axial/xd_5_3d_spgr_volume.nii.gz",
#     a)
