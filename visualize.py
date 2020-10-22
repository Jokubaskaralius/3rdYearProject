import numpy as np
import json


class Visualize:
    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
        self.loss = list()
        self.epoch = list()

    def trainingLoss(self, epoch, loss):
        self.epoch.append(epoch)
        x = self.epoch
        self.loss.append(loss)
        y = self.loss
        self.exportTrainingLoss()

    def exportTrainingLoss(self):
        # For now its okay, but maybe it's better to keep the state
        # of the data dictionary, so that we would not need to
        # iterate over the entire epoch everytime O(n^2)
        data = list()
        for i in self.epoch:
            data.append({"x": self.epoch[i], "y": self.loss[i]})

        with open('visualization/trainingLoss.json', 'w') as outfile:
            json.dump(data, outfile)
