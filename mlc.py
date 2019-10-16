"""
Multi-spectral Classification on Red & NIR Image with Maximum-likelihood Classifier
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 20
EPSILON = 1e-6


class MultiSpectralImage:
    """
    Represent a multi-spectral image
    """
    def __init__(self, red, nir, label):
        self.red = red
        self.nir = nir
        self.label = label

    def nvdi(self):
        """
        Compute NDVI values given RED and NIR image
        """
        return (self.nir - self.red) / (self.nir + self.red + EPSILON)

    @staticmethod
    def plot_hist(ndvi):
        """
        Plot histogram of NDVI values
        """
        plt.hist(np.ndarray.flatten(ndvi), bins=np.arange(np.min(ndvi), np.max(ndvi) + EPSILON, 0.2))
        plt.xlim(np.min(ndvi), np.max(ndvi))
        plt.title("histogram")
        plt.show()


class MaximumLikelihoodClassifier:
    """
    Maximum-likelihood Classifier
    """
    def __init__(self, msimage):
        self.msimge = msimage

    def learn(self):
        pass


def load_data(csv_path):
    """
    Load data from a csv file
    """
    records = []
    with open(csv_path) as data:
        reader = csv.reader(data)
        for record in reader:
            records.append(record)
    return np.array(records).astype(np.int)


if __name__ == '__main__':
    msimage = MultiSpectralImage(load_data("data/RED.csv"), load_data("data/NIR.csv"), load_data("data/label.csv"))
    ndvi = msimage.nvdi()
    msimage.plot_hist(ndvi)

