import os
import re
import math
import json
from typing import List, Dict, Any, Tuple, Optional
import random


class PathManager:
    def __init__(self, params: Dict[str, Any]):
        ''' PathManager __INIT__

        Utility component to maintain relevant project/data paths

        Input variables:
         params - dictionary, pathManager configuration parameters (config.json).
            params::data_dir_name - string, directory name where training/testing data is kept
            params::data_class_name - string, common directory name for keeping multi-class labels 
            params::modalities - list, MRI brain image modalities (e.g., T1-w, T2, FLAIR, etc.).
                                       PathManager shall include the set of modalities defined in config.json 
            params::image_extension - string, MRI brain image NIfTi format extension
        '''

        if not isinstance(params, dict):
            raise TypeError("Expected dict; got %s" % type(params).__name__)
        if not params:
            raise ValueError("Expected %s dict; got empty dict" %
                             os.path.basename(__file__))

        self.data_dir_name = params["data_dir_name"]
        if not isinstance(self.data_dir_name, str):
            raise TypeError("Expected str; got %s" %
                            type(self.data_dir_name).__name__)
        if not self.data_dir_name:
            raise ValueError("Expected %s str; got empty str" %
                             os.path.basename(__file__))

        self.visuals_dir_name = params["visuals_dir_name"]
        if not isinstance(self.visuals_dir_name, str):
            raise TypeError("Expected str; got %s" %
                            type(self.visuals_dir_name).__name__)
        if not self.visuals_dir_name:
            raise ValueError("Expected %s str; got empty str" %
                             os.path.basename(__file__))

        self.data_class_name = params["data_class_name"]
        if not isinstance(self.data_class_name, str):
            raise TypeError("Expected str; got %s" %
                            type(self.data_class_name).__name__)
        if not self.data_class_name:
            raise ValueError("Expected %s str; got empty str" %
                             os.path.basename(__file__))

        self.modalities = params["modalities"]
        if not isinstance(self.modalities, list):
            raise TypeError("Expected list; got %s" %
                            type(data_class_name).__name__)
        if not all(isinstance(modality, str) for modality in self.modalities):
            raise TypeError("Expected list of str; got non str list elements")
        if not self.modalities:
            raise ValueError("Expected list of str; got empty list")

        self.image_extension = params["image_extension"]
        if not isinstance(self.image_extension, str):
            raise TypeError("Expected str; got %s" %
                            type(self.image_extension).__name__)
        if not self.image_extension:
            raise ValueError("Expected %s str; got empty str" %
                             os.path.basename(__file__))

        self.proc_append_str = params["proc_append_str"]
        if not isinstance(self.proc_append_str, str):
            raise TypeError("Expected str; got %s" %
                            type(self.proc_append_str).__name__)
        if not self.proc_append_str:
            raise ValueError("Expected %s str; got empty str" %
                             os.path.basename(__file__))

    def root_dir(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    def data_dir(self) -> str:
        return os.path.join(self.root_dir(), self.data_dir_name)

    def visuals_dir(self) -> str:
        return os.path.join(self.root_dir(), self.visuals_dir_name)

    def visuals_data_dir(self) -> str:
        return os.path.join(self.visuals_dir(), self.data_dir_name)

    def data_class_paths(self) -> List[str]:
        '''data_class_paths

        Inputs:
        -------

        Outputs:
        -------
        data_class_list - list of paths to multi-class directories that share a
                          data_class_name defined in config.json.
        '''
        data_class_list = []
        data_subfolders = [
            f.path for f in os.scandir(self.data_dir()) if f.is_dir()
        ]
        if not data_subfolders:
            raise ValueError("Expected %s; got %s empty" %
                             (self.data_dir(), self.data_dir()))
        for subfolder in data_subfolders:
            is_match = re.search(self.data_class_name + "[0-9]$", subfolder)
            if is_match:
                data_class_list.append(subfolder)

        return data_class_list

    def class_patient_paths(self) -> Dict[str, List[str]]:
        '''class_patient_paths

        Inputs:
        -------

        Outputs:
        -------
        class_patients - dictionary, class - list of patient directory path key value pairs.
        '''
        data_classes = self.data_class_paths()
        class_patients = {}

        if not data_classes:
            raise ValueError("Expected %s[1-4]; got no %s[1-4] dir in %s" %
                             (self.data_class_name, self.data_class_name(),
                              self.data_dir()))
        for i, data_class in enumerate(data_classes):
            patient_dirs = [
                f.path for f in os.scandir(data_class) if f.is_dir()
            ]
            class_patients["%s%d" %
                           (self.data_class_name, i + 1)] = patient_dirs

        return class_patients

    def patient_modality_paths(self) -> Dict[str, List[str]]:
        '''patient_modality_paths

        Inputs:
        -------

        Outputs:
        -------
        patient_modalities - dictionary, patient directory path - list of modalities of the patient key value pairs.
        '''
        patients = self.class_patient_paths()
        patient_modalities = {}
        if not patients:
            raise ValueError(
                "Expected patients list; got no %s[1-4] patients" %
                self.data_class_name)

        for data_class in patients:
            for patient in patients[data_class]:
                modalities = [
                    f.path for f in os.scandir(patient) if f.is_dir()
                ]
                patient_modalities[patient] = modalities

        return patient_modalities

    def image_paths(self) -> List[str]:
        '''class_patient_paths

        Inputs:
        -------

        Outputs:
        -------
        image_path_list - list, all paths (processed/unprocessed) of all 
                          configured image modalities 
                          of every patient of every brain tumour grade.
        '''
        def _is_modality_match(basename: str, modalities: List[str]):
            for param_modality in modalities:
                if basename == param_modality:
                    return True
            return False

        patient_modality_paths = self.patient_modality_paths()
        image_path_list = []

        for patient in patient_modality_paths:
            for modality_path in patient_modality_paths[patient]:
                basename = os.path.basename(modality_path)
                if (_is_modality_match(basename, self.modalities)):
                    image_paths = [
                        os.path.join(modality_path, f)
                        for f in os.listdir(modality_path)
                        if os.path.isfile(os.path.join(modality_path, f))
                    ]
                    for image_path in image_paths:
                        image_path_list.append(image_path)

        return image_path_list

    def proc_image_paths(self) -> List[str]:
        '''proc_image_paths

        Inputs:
        -------

        Outputs:
        -------
        image_paths - list, processed image paths of all configured image modalities 
                      of every patient of every brain tumour grade.
        '''
        image_paths = self.image_paths()
        for image_path in self.image_paths():
            is_match = re.search(
                self.proc_append_str + self.image_extension + "$", image_path)
            if (not is_match):
                image_paths.remove(image_path)

        return image_paths

    def unproc_image_paths(self):
        '''unproc_image_paths

        Inputs:
        -------

        Outputs:
        -------
        image_paths - list, unprocessed image paths of all configured image modalities 
                      of every patient of every brain tumour grade.
        '''
        image_paths = self.image_paths()
        for image_path in self.image_paths():
            is_match = re.search(
                self.proc_append_str + self.image_extension + "$", image_path)
            if (is_match):
                image_paths.remove(image_path)

        return image_paths

    def split_path(self, path: str) -> Tuple[str, str, str]:
        '''split_path

        Inputs:
        -------
        path - str, path to a NifTI image

        Outputs:
        -------
        path - str, original path to a NifTI image
        basename - str, basename of the path to a NifTI image (no extension)
        ext - str, NifTI image file extension 
        '''
        regex_pattern = "^(.*?)" + self.image_extension
        basename = re.match(regex_pattern, path)[1]
        if not basename:
            raise ValueError(
                "Expected regex pattern %s match; got no match for %s" %
                (regex_pattern, path))
        ext = self.image_extension
        return path, basename, ext

    def append_proc_path(self, path: str) -> str:
        '''append_proc_path

        Inputs:
        -------
        path - str, path to a NifTI image

        Outputs:
        -------
        path - str, path to a NifTI image with an appended string that
               indicates the pre-processed NifTI image.
        '''
        path, basename, ext = self.split_path(path)
        basename = basename + self.proc_append_str
        path = os.path.join(path, basename + ext)
        return path

    def paths_shuffle(self,
                      image_paths: List[str],
                      shuffleSeed: Optional[int] = None) -> List[str]:
        random.seed(shuffleSeed)
        random.shuffle(image_paths)
        return image_paths


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def to_bool(value):
    valid = {
        'true': True,
        't': True,
        '1': True,
        'false': False,
        'f': False,
        '0': False,
    }
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError('invalid literal for boolean. Not a string.')

    lower_value = value.lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError('invalid literal for boolean: "%s"' % value)
    return


def export_JSON(data: Dict, path: str):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)
    return


def load_JSON(path: str):
    with open(path) as json_file:
        data = json.load(json_file)
    return data
