from features.features_factory_base import FeaturesFactory
import numpy as np


class ColorHistogramFeaturesFactory(FeaturesFactory):

    def __init__(self, n_bins=32, bins_range=(0, 256)):
        self.n_bins = n_bins
        self.bins_range = bins_range

    def create(self, image):
        assert len(image.shape) == 3 and image.shape[2] == 3, "image expected to have 3 dimensions"

        channel1_hist = np.histogram(image[:, :, 0], bins=self.n_bins, range=self.bins_range)
        channel2_hist = np.histogram(image[:, :, 1], bins=self.n_bins, range=self.bins_range)
        channel3_hist = np.histogram(image[:, :, 2], bins=self.n_bins, range=self.bins_range)

        feature_vector = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

        return feature_vector
