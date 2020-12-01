__all__ = ['ToTensor', 'FeatureScaling', 'Crop', 'Resize', 'SkullStrip']

import math
import numpy as np
import torch
from typing import Callable, Tuple, Any, Optional
import cv2
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
        if (method == "MN"):
            self.method = method
        else:
            raise ValueError(
                f'Unexpected Feature scaling method. Mean normalization - "MN" Supported only for now.\nCurrent input: {method}'
            )

    def __call__(self) -> np.ndarray:
        if (self.method == "MN"):
            max_val = np.amax(self.sample_data)
            mean = np.mean(self.sample_data)
            sample_data_0_mean = np.subtract(self.sample_data, mean)
            sample_data = np.divide(sample_data_0_mean, max_val)
        return sample_data


class Crop():
    def __init__(self, sample_data: np.ndarray):
        self.sample_data = sample_data

    def __call__(self) -> np.ndarray:
        dims = self.sample_data.ndim
        for dim in range(dims):
            crop_dim_idx = []
            for idx, item in enumerate(np.rollaxis(self.sample_data, dim)):
                is_all_zero = not np.any(item)
                if is_all_zero:
                    crop_dim_idx.append(idx)
            sample_data = np.delete(self.sample_data, crop_dim_idx, axis=dim)
        return sample_data


class Resize():
    def __init__(self,
                 sample_data: np.ndarray,
                 shape: Optional[Tuple[int, ...]] = None):
        self.sample_data = sample_data
        self.shape = shape
        #By default reduce size by 40 %
        if (shape is None):
            dims = sample_data.ndim
            data_shape = list()
            for dim in range(dims):
                scale_rate = 0.6
                data_shape.append(int(sample_data.shape[dim] * 0.6))
            self.shape = tuple(data_shape)

    def __call__(self) -> np.ndarray:
        dim_count = self.sample_data.ndim
        if (dim_count == 3):
            resized_slices = []
            new_slices = []
            #first we need to resize invididual 2D slices
            for idx in range(self._slice_count()):
                _slice = self._slice(idx)
                _slice = self._resize_2D(_slice, self.shape[:-1])
                resized_slices.append(_slice)

            #https://www.youtube.com/watch?v=lqhMTkouBx0
            chunk_size = math.ceil(self._slice_count() / self.shape[-1])
            for slice_chunk in self._chunks(resized_slices, chunk_size):
                slice_chunk = list(map(self._mean, zip(*slice_chunk)))
                new_slices.append(slice_chunk)

            if (len(new_slices) == self.shape[-1] - 1):
                new_slices.append(new_slices[-1])
            if (len(new_slices) == self.shape[-1] - 2):
                new_slices.append(new_slices[-1])
                new_slices.append(new_slices[-1])

            if (len(new_slices) == self.shape[-1] + 2):
                new_val = list(
                    map(
                        self._mean,
                        zip(*[
                            new_slices[self.shape[-1] -
                                       1], new_slices[self.shape[-1]]
                        ])))
                del new_slices[self.shape[-1]]
                new_slices[self.shape[-1] - 1] = new_val
            if (len(new_slices) == self.shape[-1] + 2):
                new_val = list(
                    map(
                        self._mean,
                        zip(*[
                            new_slices[self.shape[-1] -
                                       1], new_slices[self.shape[-1]]
                        ])))
                del new_slices[self.shape[-1]]
                new_slices[self.shape[-1] - 1] = new_val

            sample_data = np.array(new_slices).reshape((self.shape))

        elif (dim_count == 2):
            sample_data = self._resize_2D(self.sample_data, self.shape)
        else:
            raise ValueError(
                f'Unexpected data shape. Resize of 2D and 3D supported only.\nCurrent number of dimensions: {dims}'
            )
        return sample_data

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
