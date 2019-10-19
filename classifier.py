"""
Multi-spectral Classification on Red & NIR Image with Maximum-likelihood Classifier
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from data_loader import DataLoader

IMAGE_SIZE = 20
EPSILON = 1e-6
CLASS_IDS = [0, 1, 2, 3]  # 0 is non-training data; 1 is vegetation; 2 is bare ground; 3 is water
PLOT_NDVI = False


class MultiSpectralImage:
    """
    Represent a multi-spectral image
    """
    def __init__(self, red, nir, label):
        self.red = red
        self.nir = nir
        self.label = label
        self.predicted = np.zeros(label.shape).astype(np.int)

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
        indexes = np.where(is_class == True)
        vred = self.red[is_class]
        vnir = self.nir[is_class]
        if class_id == 0:
            # return data for inference
            return np.array([vred, vnir]).T, np.array(indexes).T
        else:
            # return data for training
            return np.array([vred, vnir])

    def update_predicted(self, mlclasses, indexes):
        """
        Update the image with predicted classes
        :param mlclasses: list of scalar, class number
        :param indexes: coordinates of corresponding pixel
        """
        for index, mlclass in zip(indexes, mlclasses):
            self.predicted[index[0], index[1]] = mlclass

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
    def __init__(self):
        self.samples_veg = None
        self.samples_grd = None
        self.samples_wtr = None

    def feed_training_samples(self, training_samples):
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

        # Compute overall likelihood over classes
        likelihood_veg = math.log(determinant_veg) + mahalanobis_veg
        likelihood_grd = math.log(determinant_grd) + mahalanobis_grd
        likelihood_wtr = math.log(determinant_wtr) + mahalanobis_wtr

        # Return predicted class
        likelihoods = (likelihood_veg, likelihood_grd, likelihood_wtr)
        mlclass = likelihoods.index(min(likelihoods)) + 1
        return mlclass

    @staticmethod
    def output_prediction(msimage):
        print("Predictions: \n" + str(msimage.predicted) + "\n")
        print("where 0 is the training data, 1 is vegetation, 2 is bare ground and 3 is water.")


if __name__ == '__main__':
    # Load data
    dataloader = DataLoader()
    msimage = MultiSpectralImage(dataloader.load_data("data/RED.csv"), dataloader.load_data("data/NIR.csv"), dataloader.load_data("data/label.csv"))

    # Compute NDVI
    ndvi = msimage.nvdi()
    if PLOT_NDVI:
        msimage.plot_hist(ndvi)

    # Prepare training samples
    training_samples = []
    for class_id in CLASS_IDS[1:]:
        training_samples.append(msimage.find_class(class_id))

    # Train classifier with labeled data
    mlclassifier = MaximumLikelihoodClassifier()
    mlclassifier.feed_training_samples(training_samples)
    para_veg, para_grd, para_wtr = mlclassifier.train()

    # Prepare inference samples
    inference_sampels, indexes = msimage.find_class(class_id=0)

    # Inference with trained model
    mlclasses = []
    for sample in inference_sampels:
        mlclass = mlclassifier.classify(para_veg, para_grd, para_wtr, np.array([sample]).T)
        mlclasses.append(mlclass)

    # Update the image with predicted classes
    msimage.update_predicted(mlclasses, indexes)
    mlclassifier.output_prediction(msimage)


