#Things to consider
#1. Cropping the images, reducing some of the background pixel values. But if CNN used, than convolutional layers would fix this anyways? Done
#2. Resize the image, to remove uneccessary features.
#3. Region enhancment of the tumor.

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
import imutils

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


def image_scale(image, dim=None, plot=False):
    org_height, org_width = image.shape
    res_height, res_width = dim
    if (dim is None):
        scale_percent = 60  # percent of original size
        res_width = int(image.shape[1] * scale_percent / 100)
        res_height = int(image.shape[0] * scale_percent / 100)
        dim = (res_width, res_height)
    # resize image
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    if plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1,
                                       ncols=2,
                                       figsize=(15, 7),
                                       dpi=50,
                                       sharex=True,
                                       sharey=True)
        ax2.set_title("Resized image")
        ax1.set_title("Original image")
        ax2.imshow(resized_image, cmap='gray')
        ax1.imshow(image, cmap='gray')
        plt.show()
    return resized_image


#https://www.youtube.com/watch?v=bSyY8_rTxfs&t=247s
def image_crop(image, plot=False):
    #Convert the images to grayscale
    #grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Blur the images
    grayscale = cv2.GaussianBlur(image, (5, 5), 0)
    #Threshold the images into binary
    threshold_image = cv2.threshold(grayscale, 45, 255, cv2.THRESH_BINARY)[1]

    #Erosion and Dilation to minimize the noise
    threshold_image = cv2.erode(threshold_image, None, iterations=2)
    threshold_image = cv2.dilate(threshold_image, None, iterations=2)
    #Convert it from float32 to uint8 to get the contours
    threshold_image = np.uint8(threshold_image)
    #Detect the contours of the image
    contour = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    #Grab the contours of the image
    contour = imutils.grab_contours(contour)
    c = max(contour, key=cv2.contourArea)

    #Find the extreme points where to crop the image
    extreme_pnts_left = tuple(c[c[:, :, 0].argmin()][0])
    extreme_pnts_right = tuple(c[c[:, :, 0].argmax()][0])
    extreme_pnts_top = tuple(c[c[:, :, 1].argmin()][0])
    extreme_pnts_bottom = tuple(c[c[:, :, 1].argmax()][0])

    #Plot and test the image
    new_image = image[extreme_pnts_top[1]:extreme_pnts_bottom[1],
                      extreme_pnts_left[0]:extreme_pnts_right[0]]
    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.tick_params(axis="both",
                        which="both",
                        top=False,
                        bottom=False,
                        left=False,
                        right=False,
                        labeltop=False,
                        labelbottom=False,
                        labelleft=False,
                        labelright=False)
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.tick_params(axis="both",
                        which="both",
                        top=False,
                        bottom=False,
                        left=False,
                        right=False,
                        labeltop=False,
                        labelbottom=False,
                        labelleft=False,
                        labelright=False)
        plt.title("Cropped Image")
        plt.show()
    return new_image


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

    if (debug):
        print("Center slice shape of 1st dimension", slice_0.shape)
        print("Center slice shape of 2nd dimension", slice_1.shape)
        print("Center slice shape of 3rd dimension", slice_2.shape)

    if (visualize_original):
        show_slices([slice_0, slice_1, slice_2])
        plt.suptitle("Center slices for MRI image")
        plt.tight_layout(pad=2.0)
        plt.show()

    image = slice_2
    cropped_image = image_crop(image, plot=visualize_cropped)
    norm_image = cv2.normalize(cropped_image,
                               None,
                               alpha=0,
                               beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F)

    if (visualize_standardized):
        plt.imshow(norm_image)
        plt.show()

    scaled = image_scale(norm_image, dim=(75, 75))
    processed_img = scaled
    return processed_img


testHGG = getSingleDataExample(
    "/home/jokubas/DevWork/3rdYearProject/data/HGG/BraTS19_2013_2_1/BraTS19_2013_2_1_flair.nii.gz"
)
