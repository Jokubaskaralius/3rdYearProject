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