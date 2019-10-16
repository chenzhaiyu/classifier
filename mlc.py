"""
Multi-spectral Classification on Red & NIR Image with Maximum-likelihood Classifier
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

IMAGE_SIZE = 20
EPSILON = 1e-6
CLASS_IDS = [1, 2, 3]


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
        Compute NDVI values given RED and NIR images
        """
        return (self.nir - self.red) / (self.nir + self.red + EPSILON)

    def find_class(self, class_id):
        """
        Find pixels on RED and NIR values corresponding to specific class
        class_id: 0 is non-training data, 1 is vegetation, 2 is bare ground, 3 is water
        """
        is_class = (self.label == class_id)
        vred = self.red[is_class]
        vnir = self.nir[is_class]
        return vred, vnir

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
    def __init__(self, training_samples):
        self.training_samples = training_samples

    def fit_samples(self, class_id):
        """
        Get mean and standard deviation using MLE
        """
        # TODO: may be wrong to simply use norm.fit
        mean_red, std_red = norm.fit(training_samples[class_id - 1][0])
        mean_nir, std_nir = norm.fit(training_samples[class_id - 1][1])
        return mean_red, std_red, mean_nir, std_nir
    # TODO: get the formula, link to class


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
    # Load data
    msimage = MultiSpectralImage(load_data("data/RED.csv"), load_data("data/NIR.csv"), load_data("data/label.csv"))

    # Compute NDVI
    ndvi = msimage.nvdi()
    msimage.plot_hist(ndvi)

    # Prepare training samples
    training_samples = []
    for class_id in CLASS_IDS:
        training_samples.append(msimage.find_class(class_id))

    # Classify
    mlclassifier = MaximumLikelihoodClassifier(training_samples)
    for class_id in CLASS_IDS:
        mlclassifier.fit_samples(class_id)
