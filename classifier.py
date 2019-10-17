"""
Multi-spectral Classification on Red & NIR Image with Maximum-likelihood Classifier
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from data_loader import load_data

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
        return np.array([vred, vnir])

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
        self.samples_veg = training_samples[0]
        self.samples_grd = training_samples[1]
        self.samples_wtr = training_samples[2]

    def train(self):
        """
        For each class of Gaussian model get means and covariance matrices using MLE
        """
        # Class vegetation
        mean_veg = np.array([np.mean(self.samples_veg, axis=1)]).T
        cvar_veg = np.cov(self.samples_veg)  # divided by (n-1)
        para_veg = {"mean": mean_veg, "cvar": cvar_veg}

        # Class bare ground
        mean_grd = np.array([np.mean(self.samples_grd, axis=1)]).T
        cvar_grd = np.cov(self.samples_grd)
        para_grd = {"mean": mean_grd, "cvar": cvar_grd}

        # Class water
        mean_wtr = np.array([np.mean(self.samples_wtr, axis=1)]).T
        cvar_wtr = np.cov(self.samples_wtr)
        para_wtr = {"mean": mean_wtr, "cvar": cvar_wtr}

        return para_veg, para_grd, para_wtr

    @staticmethod
    def classify(para_veg, para_grd, para_wtr, inference_sample):
        """
        Classify unlabeled pixel using calculated model parameters
        :param inference_sample: one pixel of shape (2, 1)
        """
        # Compute Mahalanobis distances
        mahalanobis_veg = np.dot(np.dot((inference_sample - para_veg["mean"]).T, np.linalg.inv(para_veg["cvar"])),
                                 (inference_sample - para_veg["mean"]))
        mahalanobis_grd = np.dot(np.dot((inference_sample - para_grd["mean"]).T, np.linalg.inv(para_grd["cvar"])),
                                 (inference_sample - para_grd["mean"]))
        mahalanobis_wtr = np.dot(np.dot((inference_sample - para_wtr["mean"]).T, np.linalg.inv(para_wtr["cvar"])),
                                 (inference_sample - para_wtr["mean"]))

        # Compute determinant of Ck
        determinant_veg = np.linalg.det(para_veg["cvar"])
        determinant_grd = np.linalg.det(para_grd["cvar"])
        determinant_wtr = np.linalg.det(para_wtr["cvar"])

        # Conpute overall likelihood over classes
        likelihood_veg = math.log(determinant_veg) + mahalanobis_veg
        likelihood_grd = math.log(determinant_grd) + mahalanobis_grd
        likelihood_wtr = math.log(determinant_wtr) + mahalanobis_wtr

        likelihoods = (likelihood_veg, likelihood_grd, likelihood_wtr)
        mlclass = likelihoods.index(min(likelihoods))
        return mlclass


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

    # Train classifier with labeled data
    mlclassifier = MaximumLikelihoodClassifier(training_samples)
    para_veg, para_grd, para_wtr = mlclassifier.train()

    # Classify unlabeled pixels
    test_sample = np.array([[200.], [100.]])
    print(mlclassifier.classify(para_veg, para_grd, para_wtr, test_sample))
