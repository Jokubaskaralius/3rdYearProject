__all__ = ['ToTensor', 'FeatureScaling', 'Crop', 'Resize', 'SkullStrip']

import math
import numpy as np
import torch
from typing import Callable, Tuple, Any, Optional
import cv2
import skimage
from deepbrain import Extractor
from multiprocessing import Pool


class ToTensor():
    def __init__(self, sample_data: np.ndarray):
        self.sample_data = sample_data

    def __call__(self) -> torch.Tensor:
        sample_data = torch.from_numpy(self.sample_data)
        return sample_data


class FeatureScaling():
    def __init__(self, sample_data: np.ndarray, method: Optional[str] = "MN"):
        self.sample_data = sample_data
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
    def __init__(self, sample_data: np.ndarray):
        self.sample_data = sample_data

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
    def __init__(self, sample_data: np.ndarray, shape: Optional[Tuple[int,
                                                                      ...]]):
        self.sample_data = sample_data
        self.shape = shape

    def __call__(self) -> np.ndarray:
        dims = self.sample_data.ndim
        print(self.sample_data.shape)
        self.sample_data = skimage.transform.resize(self.sample_data,
                                                    self.shape,
                                                    order=dims)
        print(self.sample_data.shape)
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


class SkullStrip():
    def __init__(self, sample_data: np.ndarray):
        self.sample_data = sample_data

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
