from sklearn.decomposition import PCA


#Crop a 2D image. Removes any rows or columns that are all 0 (all black)
#image - a 2D numpy array
#returns a cropped 2D numpy array
#Issue is that if it is used for cropping multiple images, It will return
#different resolutions for each image. Which is not okay for any ML algorithm.
def crop_2D(image):
    if (type(image).__module__ != np.__name__):
        print("Function crop_2D failed. Invalid argument. Pass a numpy array")
        return -1
    if (image.ndim != 2):
        print(
            "Function crop_2D failed. Invalid dimension array. Only 2D array possible."
        )
        return -1
    all_column_crop_idx = list()
    all_row_crop_idx = list()
    for idx, column in enumerate(image.T):
        is_all_zero = not np.any(column)
        if is_all_zero:
            #If all elements of the column of the image are 0, then crop the column of the image
            all_column_crop_idx.append(idx)

    column_cropped_image = np.delete(image, all_column_crop_idx, axis=1)

    for idx, row in enumerate(image):
        is_all_zero = not np.any(row)
        if is_all_zero:
            #If all elements of the row of the image are 0, then crop the row of the image
            all_row_crop_idx.append(idx)

    cropped_image = np.delete(column_cropped_image, all_row_crop_idx, axis=0)
    return cropped_image


#Modified version of the 2D crop.
#Crop any numpy array (In our case crop either a 2D voxel slice image or a 3D voxel). Removes any dimension row that is all 0.
#image - any numpy array
#returns a cropped numpy array
#Does not work as feature matrix has to remain the same. Meaning that cropping
#must work differently
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
    #print("Path:", imgPathName)

    center_x = round(data.shape[0] / 2)
    center_y = round(data.shape[1] / 2)
    center_z = round(data.shape[2] / 2)

    if (debug):
        print("Voxel shape", data.shape)

    slice_0 = data[center_x, :, :]
    slice_1 = data[:, center_y, :]
    slice_2 = data[:, :, center_z]

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
    #cropped_image = image_crop(image, plot=visualize_cropped)
    cropped_image = crop(image)

    if visualize_cropped:
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
        plt.imshow(cropped_image)
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

    norm_image = cv2.normalize(cropped_image,
                               None,
                               alpha=0,
                               beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F)

    scaled = image_scale(norm_image, dim=(150, 150))
    processed_img = scaled

    if (visualize_standardized):
        plt.imshow(processed_img)
        plt.show()

    return processed_img


# path = "_5_3d_spgr_volume.nii.gz"
# testHGG = getSingleDataExample(
#     "/home/jokubas/DevWork/3rdYearProject/data/grade1/00002/T1-axial/" + path)

##############
#Previously the classifier was designed to classify between malignant and bening tumours
#In other words high or low grade tumours
#However, this has been changed to classification by grade
##############
#Previously accuracy calculated  #accuracy = (tp + tn) / (tp + tn + fp + fn)
#However, now it is only measured by checking how many algorithm got wrong
#As there is no more a binary classification
##############
#Previously #loss_func = torch.nn.BCELoss(reduction="mean") loss was used.
#Need to look if Cross Entropy loss calculates the mean loss as well

# def validate(labels_predicted, true_labels, arr, p_thresh=0.5):
#     tp, tn, fp, fn = arr[0], arr[1], arr[2], arr[3]
#     predicted_class = (labels_predicted >= p_thresh).long()
#     arr = predicted_class.T.eq(true_labels)[0]
#     for idx, item in enumerate(arr):
#         label = int(true_labels[idx].item())
#         item = bool(item.item())
#         if (item is False and label == 0):
#             fp = fp + 1
#         elif (item is False and label == 1):
#             fn = fn + 1
#         elif (item is True and label == 1):
#             tp = tp + 1
#         else:
#             tn = tn + 1
#     return tp, tn, fp, fn

# class Transforms():
#     def __init__(self):
#         pass

#     # def to_tensor(self, sample_data: np.ndarray) -> torch.Tensor:
#     #     sample_data = torch.from_numpy(sample_data)
#     #     return sample_data

#     # def crop(self, sample_data: np.ndarray) -> np.ndarray:
#     #     dims = sample_data.ndim
#     #     for dim in range(dims):
#     #         crop_dim_idx = list()
#     #         for idx, item in enumerate(np.rollaxis(sample_data, dim)):
#     #             is_all_zero = not np.any(item)
#     #             if is_all_zero:
#     #                 crop_dim_idx.append(idx)
#     #         dim_cropped = np.delete(sample_data, crop_dim_idx, axis=dim)
#     #         sample_data = dim_cropped
#     #     return sample_data

#     def mean_normalize(self, sample_data: np.ndarray) -> np.ndarray:
#         maxVal = np.amax(sample_data)
#         mean = np.mean(sample_data)
#         sample_data_0_mean = np.subtract(sample_data, mean)
#         sample_data = np.divide(sample_data_0_mean, maxVal)
#         return sample_data

#this is 2D resize, need to fix it to apply to 3D as well.
#https://stackoverflow.com/questions/42451217/resize-3d-stack-of-images-in-python
#Not working
# def resize(self, sample_data: np.ndarray, shape=None) -> np.ndarray:
#     print(sample_data.shape)
#     data_shape = shape
#     dims = sample_data.ndim
#     if (shape is None):
#         data_shape = list()
#         for dim in range(dims):
#             scale_percent = 60
#             data_shape.append(
#                 int(sample_data.shape[dim] * scale_percent / 100))
#         data_shape = tuple(data_shape)
#     # resize image
#     print(data_shape)
#     return sample_data

# def skull_strip(self, sample_data: np.ndarray) -> np.ndarray:
#     with Pool(1) as p:
#         prob = p.apply(SkullStripProc(sample_data), ())
#         p.close()
#         p.join()
#     mask = prob > 0.5
#     mask = mask.astype(dtype=int)
#     sample_data = sample_data * mask
#     return sample_data
# class SkullStripProc():
#     def __init__(self, sample_data: np.ndarray):
#         self.sample_data = sample_data

#     def __call__(self) -> np.ndarray:
#         ext = Extractor()
#         prob = ext.run(self.sample_data)
#         return prob

# 60% training, 20% validation, 20% test
# Need to ensure that the training set has enough LGG data for classification?
# Also I shuffle the paths always. This might not be desirable,
# Because It is not possible to reproduce.
# def createPartition():
#     partition = dict()
#     imagePaths = getImagePaths(shuffle="yes", shuffleSeed=SEED)

#     trainingDatasetCount = round(len(imagePaths) * 0.6)
#     trainingDatasetPaths = imagePaths[:trainingDatasetCount]
#     for path in trainingDatasetPaths:
#         imagePaths.remove(path)

#     validationDatasetCount = round(len(imagePaths) * 0.5)
#     validationDatasetPaths = imagePaths[:validationDatasetCount]

#     testDatasetCount = validationDatasetCount
#     testDatasetPaths = imagePaths[testDatasetCount:]

#     partition["train"] = trainingDatasetPaths
#     partition["validation"] = validationDatasetPaths
#     partition["test"] = testDatasetPaths
#     return partition

# def createLabels():
#     labels = dict()
#     classes = ('grade1', 'grade2', 'grade3', 'grade4')

#     imagePaths = getImagePaths()
#     if (imagePaths == -1):
#         print("createLabels failed. imagePaths returned error status")
#         return -1

#     for path in imagePaths:
#         dataClass = re.findall("grade[1-9]", path)[0]
#         if dataClass == "grade1":
#             labels[path] = torch.tensor([1, 0, 0, 0])
#         elif dataClass == "grade2":
#             labels[path] = torch.tensor([0, 1, 0, 0])
#         elif dataClass == "grade3":
#             labels[path] = torch.tensor([0, 0, 1, 0])
#         elif dataClass == "grade4":
#             labels[path] = torch.tensor([0, 0, 0, 1])
#         else:
#             print("createLabel. No such class exists. HGG or LGG")
#             return -1
#     return labels

# def getDataClassPath():
#     dataPath = getDataPath()
#     HGGPath = os.path.join(dataPath, "HGG")
#     LGGPath = os.path.join(dataPath, "LGG")
#     if (os.path.isdir(HGGPath) is False or os.path.isdir(LGGPath) is False):
#         print("HGG or LGG not found in projectFolder/data/")
#         return -1
#     return [HGGPath, LGGPath]

# #Get HGG or LGG NIFTI image archive paths and sort them in lists
# #Flair, seg, t1, t1ce and t2 in seperate lists
# #pathName - absolute or relative path to HGG or LGG folder
# #MRIsequence - possible MRI sequences. "all" - returns all sequences, t1 returns only t1.
# #possible sequences: all, flair, seg, t1, t1ce, t2
# #returns either all lists or a certain list
# def getImagePaths(MRIsequence="all", shuffle="no", shuffleSeed=None):
#     if (getDataClassPath == -1):
#         print("Failed getDataClassPath")
#         return -1

#     flair = list()
#     seg = list()
#     t1 = list()
#     t1ce = list()
#     t2 = list()
#     for dataClassPath in getDataClassPath():
#         for idx, x in enumerate(os.walk(dataClassPath)):

#             #Skip the parent directory given by os.walk in first iteration
#             if (idx == 0):
#                 continue
#             imageFolder = x[0]
#             files = sorted([
#                 f for f in os.listdir(imageFolder)
#                 if os.path.isfile(os.path.join(imageFolder, f))
#             ])
#             flair.append(os.path.join(imageFolder, files[0]))
#             seg.append(os.path.join(imageFolder, files[1]))
#             t1.append(os.path.join(imageFolder, files[2]))
#             t1ce.append(os.path.join(imageFolder, files[3]))
#             t2.append(os.path.join(imageFolder, files[4]))

#     if (shuffleSeed == None or isinstance(shuffleSeed, int)):
#         random.seed(shuffleSeed)

#     if (shuffle == "yes" or shuffle == "Yes" or shuffle == "Y"
#             or shuffle == "y"):
#         random.shuffle(flair)
#         random.shuffle(seg)
#         random.shuffle(t1)
#         random.shuffle(t1ce)
#         random.shuffle(t2)

#     if (MRIsequence == "all"):
#         return (flair, seg, t1, t1ce, t2)
#     elif (MRIsequence == "flair" or MRIsequence == "Flair"):
#         return flair
#     elif (MRIsequence == "seg" or MRIsequence == "Seg"):
#         return seg
#     elif (MRIsequence == "t1" or MRIsequence == "T1"):
#         return t1
#     elif (MRIsequence == "t1ce" or MRIsequence == "T1ce"
#           or MRIsequence == "T1CE"):
#         return t1ce
#     elif (MRIsequence == "t2" or MRIsequence == "T2"):
#         return t2
#     else:
#         print(
#             "Invalid MRI sequence NIFTI image. Possible MRI sequences: flair, seg, t1, t1ce, t2"
#         )
#         return -1

# class Resize():
#     def __init__(self, sample_data: np.ndarray, shape: Optional[Tuple[int,
#                                                                       ...]]):
#         self.sample_data = sample_data
#         self.shape = shape

#     def __call__(self) -> np.ndarray:
#         dim_count = self.sample_data.ndim
#         if (dim_count == 3):
#             resized_slices = []
#             new_slices = []
#             #first we need to resize invididual 2D slices
#             for idx in range(self._slice_count()):
#                 _slice = self._slice(idx)
#                 _slice = self._resize_2D(_slice, self.shape[:-1])
#                 resized_slices.append(_slice)

#             #https://www.youtube.com/watch?v=lqhMTkouBx0
#             chunk_size = math.ceil(self._slice_count() / self.shape[-1])
#             for slice_chunk in self._chunks(resized_slices, chunk_size):
#                 slice_chunk = list(map(self._mean, zip(*slice_chunk)))
#                 new_slices.append(slice_chunk)

#             if (len(new_slices) == self.shape[-1] - 1):
#                 new_slices.append(new_slices[-1])
#             if (len(new_slices) == self.shape[-1] - 2):
#                 new_slices.append(new_slices[-1])
#                 new_slices.append(new_slices[-1])

#             if (len(new_slices) == self.shape[-1] + 2):
#                 new_val = list(
#                     map(
#                         self._mean,
#                         zip(*[
#                             new_slices[self.shape[-1] -
#                                        1], new_slices[self.shape[-1]]
#                         ])))
#                 del new_slices[self.shape[-1]]
#                 new_slices[self.shape[-1] - 1] = new_val
#             if (len(new_slices) == self.shape[-1] + 2):
#                 new_val = list(
#                     map(
#                         self._mean,
#                         zip(*[
#                             new_slices[self.shape[-1] -
#                                        1], new_slices[self.shape[-1]]
#                         ])))
#                 del new_slices[self.shape[-1]]
#                 new_slices[self.shape[-1] - 1] = new_val

#             sample_data = np.array(new_slices).reshape((self.shape))

#         elif (dim_count == 2):
#             sample_data = self._resize_2D(self.sample_data, self.shape)
#         else:
#             raise ValueError(
#                 f'Unexpected data shape. Resize of 2D and 3D supported only.\nCurrent number of dimensions: {dims}'
#             )
#         return sample_data

#     def _chunks(self, l: list, n: int):
#         for i in range(0, len(l), n):
#             yield l[i:i + n]

#     def _mean(self, l: list) -> int:
#         return sum(l) / len(l)

#     def _slice(self, idx: int) -> np.ndarray:
#         return self.sample_data[:, :, idx]

#     def _slice_count(self) -> int:
#         return self.sample_data.shape[-1]

#     def _resize_2D(self, sample_data, shape) -> np.ndarray:
#         sample_data = cv2.resize(sample_data,
#                                  shape,
#                                  interpolation=cv2.INTER_AREA)
#         return sample_data
