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
