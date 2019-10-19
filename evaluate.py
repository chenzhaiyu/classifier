import csv
import numpy as np
from sklearn.metrics import confusion_matrix


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
        self.confusion_matrix = None
        self.overall_accuracy = None
        self.kappa_coefficient = None

    def compute_confusion_matrix(self):
        self.confusion_matrix = confusion_matrix(self.labels, self.predictions, labels=[1, 2, 3, 4, 5]).T

    def compute_overall_accuracy(self):
        self.overall_accuracy = np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)

    def compute_kappa(self):
        N = np.sum(self.confusion_matrix)
        Sm = np.trace(self.confusion_matrix)
        Sg = np.sum([np.sum(self.confusion_matrix[i, :]) * np.sum(self.confusion_matrix[:, i]) for i in range(len(self.confusion_matrix))])
        self.kappa_coefficient = (N * Sm - Sg) / (N ** 2 - Sg)

    def output_evaluation(self):
        print("Confusion Matrix: \n" + str(self.confusion_matrix) + "\n")
        print("Overall Accuracy: " + str(self.overall_accuracy) + "\n")
        print("Kappa Coefficient: " + str(self.kappa_coefficient))


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
    evaluator.compute_confusion_matrix()
    evaluator.compute_overall_accuracy()
    evaluator.compute_kappa()
    evaluator.output_evaluation()
