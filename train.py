import os
import numpy as np
import torch
from torch import optim
from classes import Dataset, LogisticRegression

projectPathName = os.path.dirname(os.path.abspath(__file__))
dataPathName = os.path.join(projectPathName, "data")
HGGPathName = os.path.join(dataPathName, "HGG")
LGGPathName = os.path.join(dataPathName, "LGG")

if (os.path.isdir(dataPathName) is False):
    sys.exit("./data folder does not exist within project folder.")
if (os.path.isdir(HGGPathName) is False
        or os.path.isdir(LGGPathName) is False):
    sys.exit("Incorrect format of the project /data folder.")


#Get HGG or LGG NIFTI image archive paths and sort them in lists
#Flair, seg, t1, t1ce and t2 in seperate lists
#pathName - absolute or relative path to HGG or LGG folder
#MRIsequence - possible MRI sequences. "all" - returns all sequences, t1 returns only t1.
#possible sequences: all, flair, seg, t1, t1ce, t2
#returns either all lists or a certain list
def getImagePaths(pathName, MRIsequence="all"):
    if (pathName != HGGPathName and pathName != LGGPathName):
        print("Invalid path. Give the path to HGG or LGG data.")
        return -1

    flair = list()
    seg = list()
    t1 = list()
    t1ce = list()
    t2 = list()

    for idx, x in enumerate(os.walk(pathName)):
        #Skip the parent directory given by os.walk in first iteration
        if (idx == 0):
            continue
        imageFolder = x[0]
        files = sorted([
            f for f in os.listdir(imageFolder)
            if os.path.isfile(os.path.join(imageFolder, f))
        ])
        flair.append(os.path.join(imageFolder, files[0]))
        seg.append(os.path.join(imageFolder, files[1]))
        t1.append(os.path.join(imageFolder, files[2]))
        t1ce.append(os.path.join(imageFolder, files[3]))
        t2.append(os.path.join(imageFolder, files[4]))

    if (MRIsequence == "all"):
        return (flair, seg, t1, t1ce, t2)
    elif (MRIsequence == "flair" or MRIsequence == "Flair"):
        return flair
    elif (MRIsequence == "seg" or MRIsequence == "Seg"):
        return seg
    elif (MRIsequence == "t1" or MRIsequence == "T1"):
        return t1
    elif (MRIsequence == "t1ce" or MRIsequence == "T1ce"
          or MRIsequence == "T1CE"):
        return t1ce
    elif (MRIsequence == "t2" or MRIsequence == "T2"):
        return t2
    else:
        print(
            "Invalid MRI sequence NIFTI image. Possible MRI sequences: flair, seg, t1, t1ce, t2"
        )
        return -1


def createPartition():
    partition = dict()

    HGGimagePaths = getImagePaths(HGGPathName, "flair")
    if (HGGimagePaths == -1):
        print("Error occured while calling getImagePaths")
    LGGimagePaths = getImagePaths(LGGPathName, "flair")
    if (LGGimagePaths == -1):
        print("Error occured while calling getImagePaths")

    # 60 to 40 training test set for now.
    HGGtrainCount = round(len(HGGimagePaths) * 0.6)
    LGGtrainCount = round(len(LGGimagePaths) * 0.6)

    HGGTrainImagePaths = HGGimagePaths[:HGGtrainCount]
    LGGTrainImagePaths = LGGimagePaths[:LGGtrainCount]
    trainImagePaths = list()

    for item in HGGTrainImagePaths:
        trainImagePaths.append(item)
    for item in LGGTrainImagePaths:
        trainImagePaths.append(item)

    HGGTestImagePaths = HGGimagePaths[HGGtrainCount:]
    LGGTestImagePaths = LGGimagePaths[LGGtrainCount:]
    testImagePaths = list()

    for item in HGGTestImagePaths:
        testImagePaths.append(item)
    for item in LGGTestImagePaths:
        testImagePaths.append(item)

    partition["train"] = trainImagePaths
    partition["validation"] = testImagePaths
    return partition


def createLabels():
    labels = dict()

    HGGimagePaths = getImagePaths(HGGPathName, "flair")
    for item in HGGimagePaths:
        labels[item] = 1
    LGGimagePaths = getImagePaths(LGGPathName, "flair")
    for item in LGGimagePaths:
        labels[item] = 0

    return labels


def get_model():
    lr = 0.001
    model = LogisticRegression()
    loss_func = torch.nn.BCELoss()
    return loss_func, model, optim.SGD(model.parameters(), lr=lr)


#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
def train():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    device = "cpu"
    torch.backends.cudnn.benchmark = True

    # Parameters
    params = {'batch_size': 24, 'shuffle': True, 'num_workers': 6}
    max_epochs = 100

    # Datasets
    partition = createPartition()
    labels = createLabels()

    # Generators
    training_set = Dataset(partition['train'], labels)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['validation'], labels)
    validation_generator = torch.utils.data.DataLoader(validation_set,
                                                       **params)

    # Create a model
    loss_func, model, opt = get_model()

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(
                device), local_labels.to(device)

            local_labels = local_labels.view(-1, 1).float()
            local_batch = local_batch.float()

            # Model computations
            labels_predicted = model(local_batch)
            loss = loss_func(labels_predicted, local_labels)
            loss.backward()
            opt.step()
            opt.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch+1}, loss = {loss.item():.4f}')
        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(
                    device), local_labels.to(device)

                # Model computations
                labels_predicted = model(local_batch.float())
                predicted_class = labels_predicted.round()
                acc = predicted_class.eq(local_labels).sum() / float(
                    local_labels.shape[0])

        print(f'accuracy = {acc:.4f}')


train()