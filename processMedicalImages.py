#Things to consider
#1. Cropping the images, reducing some of the background pixel values. But if CNN used, than convolutional layers would fix this anyways? Done
#2. Resize the image, to remove uneccessary features.
#3. Region enhancment of the tumor.

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

debug = 0
visualize_original = 0
visualize_cropped = 0
visualize_standardized = 0


#https://nipy.org/nibabel/coordinate_systems.html
def show_slices(slices):
    #Function to display row of image slices
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


#Standardize features (mean normalization)
#features - feature numpy array
#returns standardized numpy feature array
def featureStandardization(features):
    maxVal = np.amax(features)
    mean = np.mean(features)
    features_0_mean = np.subtract(features, mean)
    features = np.divide(features_0_mean, maxVal)
    return features


#Modified version of the 2D crop.
#Crop any numpy array (In our case crop either a 2D voxel slice image or a 3D voxel). Removes any dimension row that is all 0.
#image - any numpy array
#returns a cropped numpy array
#Does not work as feature matrix has to remain the same. Meaning that cropping
#must work differently
#https://www.youtube.com/watch?v=bSyY8_rTxfs&t=247s
def crop(arr):
    if (type(arr).__module__ != np.__name__):
        print("Function crop failed. Invalid argument. Pass a numpy array")
        return -1
    dims = arr.ndim
    for dim in range(dims):
        crop_dim_idx = list()
        for idx, item in enumerate(np.rollaxis(arr, dim)):
            is_all_zero = not np.any(item)
            if is_all_zero:
                crop_dim_idx.append(idx)
        dim_cropped = np.delete(arr, crop_dim_idx, axis=dim)
        arr = dim_cropped
    return arr


#http://iyatomi-lab.info/sites/default/files/user/IEEE%20EMBC%20Arai-Chayama.pdf
#https://srinjaypaul.github.io/3D_Convolutional_autoencoder_for_brain_volumes/
#https://github.com/srinjaypaul/3D-convolutional-autoencoder-for-fmri-volumes
def pca(arr):
    scaled_X = featureStandardization(crop(arr))
    features = scaled_X.T
    pca = PCA()
    pca.fit(features)
    pca.transform(features)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

    #get all non zero Principal component variances
    per_var = np.delete(per_var,
                        [idx for idx, var in enumerate(per_var) if var == 0])

    labels = ["PC1" + str(x) for x in range(1, len(per_var) + 1)]
    bar = plt.bar(x=range(1,
                          len(per_var) + 1),
                  height=per_var,
                  tick_label=labels)

    plt.ylabel("Percentage of Explained Variance")
    plt.xlabel("Principal Component")
    plt.title("Scree plot")
    plt.xticks(rotation=90)
    plt.show()


#Function that processes a NIFTI format image
#imgPathName - absolute or relative path to the NIFTI (.nii.gz) image archive.
#label - the label passed by a calling function to specify is High grade tumour (1) or vice versa (0).
#returns an array with the 1st element being the label and the remaining are features of the image
def getSingleDataExample(imgPathName):
    if (os.path.isfile(imgPathName) is False):
        print("File not found.")
        return -1
    #if (label is None):
    #    print("No label specified")
    #    return -1
    try:
        img = nib.load(imgPathName)
    except:
        print("Incorrect MRI image format. Supported: .nii.gz")
        return -1
    data = img.get_fdata(dtype=np.float32)

    if (debug):
        print("Voxel shape", data.shape)
        print("Cropped Voxel shape", crop(data).shape)

    slice_0 = data[120, :, :]
    slice_1 = data[:, 120, :]
    slice_2 = data[:, :, 77]
    #pca(slice_2)

    featureStandardization(slice_0)

    slice_0_cropped = crop(slice_0)
    slice_1_cropped = crop(slice_1)
    slice_2_cropped = crop(slice_2)

    if (debug):
        print("Center slice shape of 1st dimension", slice_0.shape)
        print("Center slice shape of 2nd dimension", slice_1.shape)
        print("Center slice shape of 3rd dimension", slice_2.shape)

    if (visualize_original):
        show_slices([slice_0, slice_1, slice_2])
        plt.suptitle("Center slices for MRI image")
        plt.tight_layout(pad=2.0)
        plt.show()

    if (visualize_cropped):
        show_slices([slice_0_cropped, slice_1_cropped, slice_2_cropped])
        plt.suptitle("Cropped Center slices for MRI image")
        plt.tight_layout(pad=2.0)
        plt.show()

    if (visualize_standardized):
        np.savetxt("regular.csv", slice_0.flatten(), delimiter=",")
        slice_0_stand = featureStandardization(slice_0)
        np.savetxt("standardised.csv", slice_0_stand.flatten(), delimiter=",")
        slice_1_stand = featureStandardization(slice_1)
        slice_2_stand = featureStandardization(slice_2)
        show_slices([slice_0_stand, slice_1_stand, slice_2_stand])
        plt.suptitle(
            "Standardized (mean normalized) Center slices for MRI image")
        plt.tight_layout(pad=2.0)
        plt.show()

    features = slice_2
    features = featureStandardization(features)
    return features


test = getSingleDataExample(
    "/home/jokubas/DevWork/3rdYearProject/data/HGG/BraTS19_2013_2_1/BraTS19_2013_2_1_flair.nii.gz"
)