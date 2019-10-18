import csv
import numpy as np


class DEMPoint:
    """
    Represent a point associated with 3D coordinates, ground truth class and classified class
    """
    def __init__(self, x, y, z, label, prediction):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.label = int(label)
        self.prediction = int(prediction)


class Evaluator:
    """
    An evaluator to compute the confusion matrix, overall accuracy and kappa coefficient
    """
    def __init__(self, dempoints):
        self.labels = np.array([[dempoint.label for dempoint in dempoints]]).T
        self.predictions = np.array([[dempoint.prediction for dempoint in dempoints]]).T

    def confmat(self):
        pass

    def accuracy(self):
        pass

    def kappa(self):
        pass


def load_data(csv_name):
    """
    Load data from a csv file
    """
    dempoints = []
    with open(csv_name) as data:
        reader = csv.reader(data)
        for i, record in enumerate(reader):
            dempoint = DEMPoint(record[0], record[1], record[2], record[3], record[4])
            dempoints.append(dempoint)
    return dempoints


if __name__ == '__main__':
    dempoints = load_data("data/POINTCLOUD.csv")
    evaluator = Evaluator(dempoints)
    pass
