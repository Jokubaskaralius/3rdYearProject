import numpy as np
import json


class Visualize:
    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
        self.training_loss = list()
        self.validation_loss = list()
        self.epoch = list()

    def trainingLoss(self, epoch, loss):
        self.epoch.append(epoch)
        self.training_loss.append(loss)
        self.exportTrainingLoss()

    def validationLoss(self, epoch, loss):
        self.validation_loss.append(loss)
        self.exportValidationLoss()

    def exportTrainingLoss(self):
        # For now its okay, but maybe it's better to keep the state
        # of the data dictionary, so that we would not need to
        # iterate over the entire epoch everytime O(n^2)
        data = list()
        for i in self.epoch:
            data.append({"x": self.epoch[i] + 1, "y": self.training_loss[i]})

        with open('visualization/data/trainingLoss.json', 'w') as outfile:
            json.dump(data, outfile)

    def exportValidationLoss(self):
        # For now its okay, but maybe it's better to keep the state
        # of the data dictionary, so that we would not need to
        # iterate over the entire epoch everytime O(n^2)
        data = list()
        for i in self.epoch:
            data.append({"x": self.epoch[i] + 1, "y": self.validation_loss[i]})

        with open('visualization/data/validationLoss.json', 'w') as outfile:
            json.dump(data, outfile)

    def confusionMatrix(self, tp, tn, fp, fn, epoch):
        data = dict()
        data["tp"] = tp
        data["tn"] = tn
        data["fp"] = fp
        data["fn"] = fn
        data["epoch"] = epoch
        self.exportConfusionMatrixData([data])

    def exportConfusionMatrixData(self, data):
        with open('visualization/data/confusionMatrix.json', 'w') as outfile:
            json.dump(data, outfile)
