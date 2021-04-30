__all__ = [
    'ToTensor', 'FeatureScaling', 'Crop', 'Resize', 'GaussianBlur',
    'SkullStrip', 'Registration'
]

import math
import numpy as np
import torch
from typing import Callable, Tuple, Any, Optional
import cv2
<<<<<<< HEAD
import nibabel as nib
import skimage
from skimage import filters
=======
import skimage
>>>>>>> 31a0a968e1c33e2e3301755ae2551edaa31534c4
from deepbrain import Extractor
from multiprocessing import Pool
from dipy.align import (affine_registration, center_of_mass, translation,
                        rigid, affine, register_dwi_to_template)
from dipy.viz import regtools


class TransformManager:
    def __init__(self, config):
        self.transform_str = config["transforms"]
        self.transforms_map = {
            "ToTensor": ToTensor,
            "FeatureScaling": FeatureScaling,
            "Crop": Crop,
            "Resize": Resize,
            "GaussianBlur": GaussianBlur,
            "SkullStrip": SkullStrip,
            "Registration": Registration
        }

    def transforms(self):
        transforms = list()
        for transform in self.transform_str:
            transform_name = transform[0]
            transform_args = transform[1]
            transform_func = self.transforms_map[transform_name]
            transforms.append([transform_func, transform_args])
        return transforms


class ToTensor():
    def __init__(self, sample: Tuple):
        self.sample_data, self.sample_affine, self.sample_header = sample

    def __call__(self) -> torch.Tensor:
        sample_data = torch.from_numpy(self.sample_data)
        return sample_data


class FeatureScaling():
    def __init__(self, sample: Tuple, method: Optional[str] = "MN"):
        self.sample_data, self.sample_affine, self.sample_header = sample
        if (method == "MN" or method == "ZSN" or method == "MM"):
            self.method = method
        else:
            raise ValueError(
                f'''Unexpected Feature scaling method. Mean normalization - "MN" Supported only for now.
                \nCurrent input: {method}''')

    def __call__(self) -> np.ndarray:
        if (self.method == "MM"):
            max_val = np.amax(self.sample_data)
            min_val = np.amin(self.sample_data)
            sample_data_temp = np.subtract(self.sample_data, min_val)
            sample_data = np.divide(sample_data_temp, max_val - min_val)
        if (self.method == "MN"):
            max_val = np.amax(self.sample_data)
            min_val = np.amin(self.sample_data)
            mean = np.mean(self.sample_data)
            sample_data_0_mean = np.subtract(self.sample_data, mean)
            sample_data = np.divide(sample_data_0_mean, max_val - min_val)
        elif (self.method == "ZSN"):
            mean = np.mean(self.sample_data)
            std = np.std(self.sample_data)
            sample_data_0_mean = np.subtract(self.sample_data, mean)
            sample_data = np.divide(sample_data_0_mean, std)

        return sample_data


class Crop():
    def __init__(self, sample: Tuple):
        self.sample_data, self.sample_affine, self.sample_header = sample

    def __call__(self) -> np.ndarray:
        dims = self.sample_data.ndim
        crop_dim_idx = []
        for dim in range(dims):
            for idx, s in enumerate(self.sample_data):
                is_all_zero = s < 0.1
                if (np.all(is_all_zero)):
                    crop_dim_idx.append(idx)

            self.sample_data = np.delete(self.sample_data, crop_dim_idx, 0)
            crop_dim_idx.clear()

            self.sample_data = np.moveaxis(self.sample_data, 0, -1)

        return self.sample_data


class Resize():
<<<<<<< HEAD
    def __init__(self, sample: Tuple, shape: Optional[Tuple[int, ...]]):
        self.sample_data, self.sample_affine, self.sample_header = sample
=======
    def __init__(self, sample_data: np.ndarray, shape: Optional[Tuple[int,
                                                                      ...]]):
        self.sample_data = sample_data
>>>>>>> 31a0a968e1c33e2e3301755ae2551edaa31534c4
        self.shape = shape

    def __call__(self) -> np.ndarray:
        dims = self.sample_data.ndim
<<<<<<< HEAD
        self.sample_data = skimage.transform.resize(self.sample_data,
                                                    self.shape,
                                                    order=dims)

=======
        print(self.sample_data.shape)
        self.sample_data = skimage.transform.resize(self.sample_data,
                                                    self.shape,
                                                    order=dims)
        print(self.sample_data.shape)
>>>>>>> 31a0a968e1c33e2e3301755ae2551edaa31534c4
        return self.sample_data

    def _chunks(self, l: list, n: int):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def _mean(self, l: list) -> int:
        return sum(l) / len(l)

    def _slice(self, idx: int) -> np.ndarray:
        return self.sample_data[:, :, idx]

    def _slice_count(self) -> int:
        return self.sample_data.shape[-1]

    def _resize_2D(self, sample_data, shape) -> np.ndarray:
        sample_data = cv2.resize(sample_data,
                                 shape,
                                 interpolation=cv2.INTER_AREA)
        return sample_data


class GaussianBlur():
    def __init__(self, sample: Tuple, sigma: Optional[float] = 0):
        self.sample_data, self.sample_affine, self.sample_header = sample
        self.sigma = sigma

    def __call__(self):
        # Scikit image gaussian blur
        self.sample_data = filters.gaussian(self.sample_data, sigma=self.sigma)
        return self.sample_data


class SkullStrip():
    def __init__(self, sample: Tuple):
        self.sample_data, self.sample_affine, self.sample_header = sample

    def __call__(self) -> np.ndarray:
        with Pool(1) as p:
            prob = p.apply(self._skull_strip_func, ())
            p.close()
            p.join()
        mask = prob > 0.5
        mask = mask.astype(dtype=int)
        sample_data = self.sample_data * mask
        return sample_data

    def _skull_strip_func(self) -> np.ndarray:
        ext = Extractor()
        prob = ext.run(self.sample_data)
        return prob


class Registration():
    def __init__(self, sample: Tuple, static_path: Optional[str]):
        if not isinstance(static_path, str):
            raise TypeError("Expected str; got %s" %
                            type(static_path).__name__)
        if not static_path:
            raise ValueError("Expected %s str; got empty str" %
                             os.path.basename(__file__))

        self.static_path = static_path
        self._static_sample()
        self.sample_data, self.sample_affine, self.sample_header = sample
        self.pipeline = [center_of_mass, translation, rigid, affine]

    def __call__(self):
        xformed_img, reg_affine = affine_registration(
            self.sample_data,  # moving image data
            self.static_data,  # static or template image data
            moving_affine=self.sample_affine,
            static_affine=self.static_affine,
            nbins=32,
            metric='MI',
            pipeline=self.pipeline,
            level_iters=[10000, 1000, 100],
            sigmas=[3.0, 1.0, 0.0],
            factors=[4, 2, 1])

        return xformed_img

    def _static_sample(self):
        sample = nib.load(self.static_path)
        self.static_data = sample.get_fdata(dtype=np.float32)
        self.static_header = sample.header
        #look into getting affine.
        self.static_affine = self.static_header.get_best_affine()