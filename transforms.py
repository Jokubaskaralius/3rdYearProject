import numpy as np
import torch
from typing import Callable
import cv2
from deepbrain import Extractor
from multiprocessing import Pool


class Transforms():
    def __init__(self):
        pass

    def to_tensor(self, sample_data: np.ndarray) -> torch.Tensor:
        sample_data = torch.from_numpy(sample_data)
        return sample_data

    def crop(self, sample_data: np.ndarray) -> np.ndarray:
        dims = sample_data.ndim
        for dim in range(dims):
            crop_dim_idx = list()
            for idx, item in enumerate(np.rollaxis(sample_data, dim)):
                is_all_zero = not np.any(item)
                if is_all_zero:
                    crop_dim_idx.append(idx)
            dim_cropped = np.delete(sample_data, crop_dim_idx, axis=dim)
            sample_data = dim_cropped
        return sample_data

    def mean_normalize(self, sample_data: np.ndarray) -> np.ndarray:
        maxVal = np.amax(sample_data)
        mean = np.mean(sample_data)
        sample_data_0_mean = np.subtract(sample_data, mean)
        sample_data = np.divide(sample_data_0_mean, maxVal)
        return sample_data

    #this is 2D resize, need to fix.
    #https://stackoverflow.com/questions/42451217/resize-3d-stack-of-images-in-python
    #Not working
    def resize(self, sample_data: np.ndarray, shape=None) -> np.ndarray:
        print(sample_data.shape)
        data_shape = shape
        dims = sample_data.ndim
        if (shape is None):
            data_shape = list()
            for dim in range(dims):
                scale_percent = 60
                data_shape.append(
                    int(sample_data.shape[dim] * scale_percent / 100))
            data_shape = tuple(data_shape)
        # resize image
        print(data_shape)

        # if dims == 2:
        #     sample_data = cv2.resize(sample_data,
        #                              data_shape,
        #                              interpolation=cv2.INTER_AREA)
        # elif dims == 3:
        sample_data = torch.from_numpy(sample_data)
        torch.nn.functional.interpolate(sample_data, data_shape)
        sample_data = sample_data.numpy()
        # for idx in range(data_shape[2]):
        #     img_2D = img_stack[:, :, idx]
        #     img_2D_resized = cv2.resize(img_2D,
        #                                 data_shape,
        #                                 interpolation=cv2.INTER_AREA)
        #     img_stack_sm[:, :, idx] = img_2D_resized
        # else:
        #     raise ValueError(
        #         f'Unexpected data shape. Resize of 2D and 3D supported only.\nCurrent number of dimensions: {dims}'
        #     )

        print(sample_data.shape)
        return sample_data

    def skull_strip(self, sample_data: np.ndarray) -> np.ndarray:
        with Pool(1) as p:
            prob = p.apply(SkullStripProc(sample_data), ())
            p.close()
            p.join()
        mask = prob > 0.5
        mask = mask.astype(dtype=int)
        sample_data = sample_data * mask
        return sample_data


class SkullStripProc():
    def __init__(self, sample_data: np.ndarray):
        self.sample_data = sample_data

    def __call__(self) -> np.ndarray:
        ext = Extractor()
        prob = ext.run(self.sample_data)
        return prob