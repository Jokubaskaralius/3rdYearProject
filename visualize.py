import numpy as np
import json


class Visualize:
    def __init__(self):
        self.training_loss = list()
        self.validation_loss = list()
        self.epoch = list()

    def trainingLoss(self, epoch, loss):
        # For now its okay, but maybe it's better to keep the state
        # of the data dictionary, so that we would not need to
        # iterate over the entire epoch everytime O(n^2)
        self.epoch.append(epoch)
        self.training_loss.append(loss)
        data = list()
        for i in self.epoch:
            data.append({"x": self.epoch[i] + 1, "y": self.training_loss[i]})
        self.exportJSON("trainingLoss", data)

    def validationLoss(self, epoch, loss):
        # For now its okay, but maybe it's better to keep the state
        # of the data dictionary, so that we would not need to
        # iterate over the entire epoch everytime O(n^2)
        self.validation_loss.append(loss)
        data = list()
        for i in self.epoch:
            data.append({"x": self.epoch[i] + 1, "y": self.validation_loss[i]})

        self.exportJSON("validationLoss", data)

    def confusionMatrix(self, tp, tn, fp, fn, epoch):
        data = dict()
        data["tp"] = tp
        data["tn"] = tn
        data["fp"] = fp
        data["fn"] = fn
        data["epoch"] = epoch
        self.exportJSON("confusionMatrix", [data])

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