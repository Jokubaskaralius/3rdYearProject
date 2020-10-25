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
