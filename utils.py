import os
import sys
import random
import re


def getProjectPath():
    projectPathName = os.path.dirname(os.path.abspath(__file__))
    return projectPathName


def getDataPath():
    dataPath = os.path.join(getProjectPath(), "data")
    if (os.path.isdir(dataPath) is False):
        print("./data folder does not exist within project folder.")
        return -1
    return dataPath


def getDataClassPaths():
    dataPath = getDataPath()
    grade1_path = os.path.join(dataPath, "grade1")
    grade2_path = os.path.join(dataPath, "grade2")
    grade3_path = os.path.join(dataPath, "grade3")
    grade4_path = os.path.join(dataPath, "grade4")
    if (os.path.isdir(grade1_path) is False
            or os.path.isdir(grade2_path) is False
            or os.path.isdir(grade3_path) is False
            or os.path.isdir(grade4_path) is False):
        print("Brain tumour grade data not found in projectFolder/data/")
        return -1
    return [grade1_path, grade2_path, grade3_path, grade4_path]


def getImagePaths(shuffle="no", shuffleSeed=None):
    dataClassPaths = getDataClassPaths()
    if (dataClassPaths == -1):
        print("Failed getDataClassPath")
        return -1

    t1 = list()
    for dataClassPath in dataClassPaths:
        for idx, x in enumerate(os.walk(dataClassPath)):
            if (idx == 0):
                continue
            sequenceFolderPath = re.findall("^.*T1-axial$", x[0])
            if (sequenceFolderPath):
                files = sorted([
                    f for f in os.listdir(sequenceFolderPath[0])
                    if os.path.isfile(os.path.join(sequenceFolderPath[0], f))
                ])
                t1.append(os.path.join(sequenceFolderPath[0], files[0]))

    if (shuffleSeed == None or isinstance(shuffleSeed, int)):
        random.seed(shuffleSeed)

    if (shuffle == "yes" or shuffle == "Yes" or shuffle == "Y"
            or shuffle == "y"):
        random.shuffle(t1)

    return t1


getImagePaths()

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
