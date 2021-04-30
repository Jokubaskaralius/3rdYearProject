import typing
import numpy as np
import json
import nibabel as nib
from skimage.transform import resize
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from dataset import DatasetManager
from utils import *
from transforms import *


class Visualize:
    def __init__(self):
        self.training_loss = list()
        self.validation_loss = list()
        self.epoch = list()

    def trainingLoss(self, epoch, loss):
        self.epoch.append(epoch)
        self.training_loss.append(loss)
        data = list()
        for i in self.epoch:
            data.append({"x": self.epoch[i] + 1, "y": self.training_loss[i]})
        self.exportJSON("trainingLoss", data)

    def validationLoss(self, epoch, loss):
        self.validation_loss.append(loss)
        data = list()
        for i in self.epoch:
            data.append({"x": self.epoch[i] + 1, "y": self.validation_loss[i]})

        self.exportJSON("validationLoss", data)

    def confusionMatrix(self, performance_list, epoch):
        def slice_per(source, step):
            return [source[i::step] for i in range(step)]

        matching_matrix_list = list()
        confusion_matrix_list = list()
        performance_matrix_list = list()
        for fold_performance_measures in performance_list:
            matching_matrix = list()
            for idx, row in enumerate(fold_performance_measures[0]):
                for idy, col in enumerate(row):
                    matching_matrix.append(int(col))
            matching_matrix_list.append(matching_matrix)

            confusion_matrix = list()
            for idx, row in enumerate(fold_performance_measures[1]):
                for idy, col in enumerate(row):
                    confusion_matrix.append(int(col))
            confusion_matrix_list.append(
                slice_per(confusion_matrix,
                          len(fold_performance_measures[1][0])))

            performance_matrix = list()
            for idx, row in enumerate(fold_performance_measures[2]):
                for idy, col in enumerate(row):
                    performance_matrix.append(float("{:.2f}".format(col)))

            performance_matrix_list.append(
                slice_per(performance_matrix,
                          len(fold_performance_measures[2][0])))

        self.exportJSON("confusionMatrix", [{
            "matching_matrix_list": matching_matrix_list,
            "confusion_matrix_list": confusion_matrix_list,
            "performance_matrix_list": performance_matrix_list,
            "epoch": epoch
        }])

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


#https://nipy.org/nibabel/coordinate_systems.html
def show_slices(slices):
    #Function to display row of image slices
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def visualizeImage2D():
    processed_image_path = "/home/jokubas/DevWork/3rdYearProject/data/grade1/sub-PAT08/T1-axial/sub-PAT08_ses_ch-preop_T1w_processed.nii.gz"
    unprocessed_image_path = "/home/jokubas/DevWork/3rdYearProject/data/grade1/sub-PAT08/T1-axial/sub-PAT08_ses_ch-preop_T1w.nii.gz"

<<<<<<< HEAD
    config = json.load(open("./config.json"))
    path_manager = PathManager(config["pathManager"])

    dataset_manager = DatasetManager(config["datasetManager"], path_manager)
=======
    dataset_manager = DatasetManager([
        [FeatureScaling, ["MM"]],
        [Crop, []],
        [Resize, [(100, 100, 56)]],
        #[SkullStrip, []],
        #[ToTensor, []]
    ])
>>>>>>> 31a0a968e1c33e2e3301755ae2551edaa31534c4
    dataset_manager.process_image(unprocessed_image_path)

    img_proc = nib.load(processed_image_path)
    img_unproc = nib.load(unprocessed_image_path)

    data_proc = img_proc.get_fdata()
    data_unproc = img_unproc.get_fdata()

    print("Processed image shape", data_proc.shape)
    print("Unprocessed image shape", data_unproc.shape)

<<<<<<< HEAD
    # # resizing
    # slice_0_proc = data_proc[50, :, :]
    # slice_1_proc = data_proc[:, 50, :]
    # slice_2_proc = data_proc[:, :, 28]

    # slice demo for cropping
    # slice_0_proc = data_proc[80, :, :]
    # slice_1_proc = data_proc[:, 93, :]
    # slice_2_proc = data_proc[:, :, 95]
=======
    slice_0_proc = data_proc[50, :, :]
    slice_1_proc = data_proc[:, 50, :]
    slice_2_proc = data_proc[:, :, 28]

    # slice demo for cropping
    # slice_0_proc = data_proc[83, :, :]
    # slice_1_proc = data_proc[:, 105, :]
    # slice_2_proc = data_proc[:, :, 84]
>>>>>>> 31a0a968e1c33e2e3301755ae2551edaa31534c4

    slice_0_proc = data_proc[80, :, :]
    slice_1_proc = data_proc[:, 128, :]
    slice_2_proc = data_proc[:, :, 128]

    slice_0_unproc = data_unproc[80, :, :]
    slice_1_unproc = data_unproc[:, 128, :]
    slice_2_unproc = data_unproc[:, :, 128]

    show_slices([slice_0_proc, slice_1_proc, slice_2_proc])
    plt.suptitle("Center slices for processed MRI image")
    plt.tight_layout(pad=2.0)
    plt.show()

    show_slices([slice_0_unproc, slice_1_unproc, slice_2_unproc])
    plt.suptitle("Center slices for unprocessed MRI image")
    plt.tight_layout(pad=2.0)
    plt.show()


#visualizeImage2D()


#https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
#https://stackoverflow.com/questions/14995610/how-to-make-a-4d-plot-with-matplotlib-using-arbitrary-data
def visualizeImage3D():
    processed_image_path = "/home/jokubas/DevWork/3rdYearProject/data/nifti_TCGA_LGG/TCGA-FG-A6J1/T1-axial/19_t1_mprage_ax_gd_processed.nii.gz"
    unprocessed_image_path = "/home/jokubas/DevWork/3rdYearProject/data/nifti_TCGA_LGG/TCGA-FG-A6J1/T1-axial/19_t1_mprage_ax_gd.nii.gz"

    plot = Plot3D()

    dataset_manager = DatasetManager([
        [FeatureScaling, ["MM"]],
        [Crop, []],
        [Resize, [(50, 50, 26)]],
        [GaussianBlur, []],
        [SkullStrip, []],
        #[ToTensor, []]
    ])
    dataset_manager.process_image(unprocessed_image_path)

    img_proc = nib.load(processed_image_path)
    img_unproc = nib.load(unprocessed_image_path)

    data_proc = img_proc.get_fdata()
    data_unproc = img_unproc.get_fdata()

    x = []
    y = []
    z = []
    c = []
    for sur in range(data_proc.shape[0]):
        for row in range(data_proc.shape[1]):
            for col in range(data_proc.shape[2]):
                x.append(sur)
                y.append(row)
                z.append(col)
                c.append(data_proc[sur][row][col])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Choose colormap
    cmap = pl.cm.RdBu

    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    img = ax.scatter(x, y, z, c=c, cmap=my_cmap)
    fig.colorbar(img)
    plt.show()


#visualizeImage3D()
