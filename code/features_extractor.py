import numpy as np
import matplotlib.image as mpimg
import cv2


class FeaturesExtractor:

    def __init__(self, features_providers, color_space="RGB"):
        self.features_providers = features_providers
        self.color_space = color_space

    def extract_from_image(self, rgb_image):
        image = self._change_color_space(rgb_image, self.color_space)
        image_features = [x.create(image) for x in self.features_providers]
        return np.concatenate(image_features)

    def extract_from_files(self, image_paths):
        features = []
        for image_path in image_paths:
            rgb_image = mpimg.imread(image_path)
            image_features = self.extract_from_image(rgb_image)
            features.append(image_features)

        return features

    @staticmethod
    def _change_color_space(rgb_image, color_space):
        if color_space is "RGB":
            return np.copy(rgb_image)

        from_color_space = "COLOR_RGB2"

        destination_color_space = from_color_space + color_space
        if hasattr(cv2, destination_color_space):
            destination_color_space_code = getattr(cv2, destination_color_space)
            return cv2.cvtColor(rgb_image, destination_color_space_code)
        else:
            supported_conversion_keys = filter(lambda y: y.startswith(from_color_space), dir(cv2))
            supported_color_spaces = list(map(lambda x: x.replace(from_color_space, ""), supported_conversion_keys))

            msg = "Not supported color space: '{}'. \n\tSupported color spaces: {}.".format(
                color_space,
                ", ".join(supported_color_spaces))
            raise Exception(msg)
