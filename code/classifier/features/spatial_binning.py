from features.features_factory_base import FeaturesFactory
import cv2


class SpatialBinningOfColorFeaturesFactory(FeaturesFactory):

    def __init__(self, size=(32, 32)):
        self.size = size

    def create(self, image):
        feature_vector = cv2.resize(image, self.size).ravel()
        return feature_vector

