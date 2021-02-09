import typing
import numpy as np
import json
import nibabel as nib
from skimage.transform import resize
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DatasetManager import DatasetManager
from transforms import *


class Visualize:
    def __init__(self):
        self.training_loss = list()
        self.validation_loss = list()
        self.epoch = list()

    def trainingLoss(self, epoch, loss):
        # For now its okay, but maybe it's better to keep the state
        # of the data dictionary, so that we would not need to
        # iterate over the entire epoch everytime O(n^2)
        self.epoch.append(epoch)
        self.training_loss.append(loss)
        data = list()
        for i in self.epoch:
            data.append({"x": self.epoch[i] + 1, "y": self.training_loss[i]})
        self.exportJSON("trainingLoss", data)

    def validationLoss(self, epoch, loss):
        # For now its okay, but maybe it's better to keep the state
        # of the data dictionary, so that we would not need to
        # iterate over the entire epoch everytime O(n^2)
        self.validation_loss.append(loss)
        data = list()
        for i in self.epoch:
            data.append({"x": self.epoch[i] + 1, "y": self.validation_loss[i]})

        self.exportJSON("validationLoss", data)

    def confusionMatrix(self, matching_matrix, confusion_matrix,
                        performance_matrix, epoch):
        matching_list = list()
        for idx, row in enumerate(matching_matrix):
            matching_dict = dict()
            matching_dict["m_0"] = int(row[0])
            matching_dict["m_1"] = int(row[1])
            matching_dict["m_2"] = int(row[2])
            matching_dict["m_3"] = int(row[3])
            matching_list.append(matching_dict)

        confusion_list = list()
        # for idx, row in enumerate(confusion_matrix):
        #     confusion_dict = dict()
        #     confusion_dict["tp"] = int(row[0])
        #     confusion_dict["tn"] = int(row[1])
        #     confusion_dict["fp"] = int(row[2])
        #     confusion_dict["fn"] = int(row[3])
        #     confusion_list.append(confusion_dict)

        performance_list = list()
        for idx, row in enumerate(performance_matrix):
            performance_dict = dict()
            performance_dict["accuracy"] = float(row[0])
            performance_dict["precision"] = float(row[1])
            performance_dict["recall"] = float(row[2])
            performance_dict["f1"] = float(row[3])
            performance_list.append(performance_dict)

        self.exportJSON("confusionMatrix",
                        [matching_list, performance_list, epoch])

    def ROC(self, data):
        data_roc = list()
        for item in data:
            threshold = item[0]
            true_positive_rate = item[1]
            false_positive_rate = item[2]
            obj = [{
                "x": false_positive_rate,
                "y": true_positive_rate
            }, threshold]
            data_roc.append(obj)
        self.exportJSON("ROC", data_roc)

    def exportJSON(self, filename, data):
        pathname = "visualization/data/" + filename + ".json"
        with open(pathname, 'w') as outfile:
            json.dump(data, outfile)


##############################################################


class Plot3D:
    def __init__(self):
        pass

    def normalize(self, arr):
        arr_min = np.min(arr)
        return (arr - arr_min) / (np.max(arr) - arr_min)

    def explode(self, data):
        shape_arr = np.array(data.shape)
        size = shape_arr[:3] * 2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]),
                            dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    def expand_coordinates(self, indices):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z

    def plot_cube(self, cube, ax, angle=320):
        cube = self.normalize(cube)
        facecolors = cm.viridis(cube)
        facecolors[:, :, :, -1] = cube
        facecolors = self.explode(facecolors)

        filled = facecolors[:, :, :, -1] != 0
        x, y, z = self.expand_coordinates(
            np.indices(np.array(filled.shape) + 1))

        ax.view_init(30, angle)
        ax.set_xlim(right=cube.shape[0] * 2)
        ax.set_ylim(top=cube.shape[1] * 2)
        ax.set_zlim(top=cube.shape[2] * 2)

        ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)

        return ax


def dataset_plot_3D():
    dataset_manager = DatasetManager()
    plot = Plot3D()

    processed_paths = dataset_manager.processed_image_paths()
    unprocessed_paths = dataset_manager.unprocessed_image_paths()

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    for i in range(len(processed_paths)):
        processed_img = nib.load(processed_paths[i])
        processed_data = processed_img.get_fdata()
        #processed_data = resize(processed_data, (50, 50, 10), mode='constant')
        unprocessed_img = nib.load(unprocessed_paths[i])
        unprocessed_data = unprocessed_img.get_fdata()
        unprocessed_data = resize(unprocessed_data, (50, 50, 10),
                                  mode='constant')

        processed_ax = plot.plot_cube(processed_data, ax1)
        unprocessed_ax = plot.plot_cube(unprocessed_data, ax2)

        plt.show()


#dataset_plot_3D()


#https://terbium.io/2017/12/matplotlib-3d/
def pixel_intensity_histogram(processed_data=None, unprocessed_data=None):
    fig, axes = plt.subplots(1, 2)

    if (unprocessed_data is not None):
        n, bins, patches = axes[0].hist(unprocessed_data.reshape(-1),
                                        50,
                                        density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # for c, p in zip(bin_centers, patches):
        #     axes[0].setp(p, 'facecolor', cm.viridis(c))

    if (processed_data is not None):
        n, bins, patches = axes[1].hist(processed_data.reshape(-1),
                                        50,
                                        density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # for c, p in zip(bin_centers, patches):
        #     axes[1].setp(p, 'facecolor', cm.viridis(c))

    if (unprocessed_data is not None or processed_data is not None):
        plt.show()


def dataset_pixel_intensity():
    dataset_manager = DatasetManager()

    processed_paths = dataset_manager.processed_image_paths()
    unprocessed_paths = dataset_manager.unprocessed_image_paths()

    for i in range(len(processed_paths)):
        processed_img = nib.load(processed_paths[i])
        processed_data = processed_img.get_fdata()
        unprocessed_img = nib.load(unprocessed_paths[i])
        unprocessed_data = unprocessed_img.get_fdata()

        pixel_intensity_histogram(processed_data=processed_data,
                                  unprocessed_data=unprocessed_data)


def debug():
    dataset_manager = DatasetManager([[Resize, [(50, 50, 10)]]])
    dataset_manager.process_images()
    dataset_plot_3D()


#debug()