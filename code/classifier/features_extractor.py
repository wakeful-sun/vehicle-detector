import numpy as np
import cv2


class FeaturesExtractor:

    def __init__(self, features_providers, color_space="BGR", image_size=(64, 64)):
        self.features_providers = features_providers
        self.color_space = color_space
        self.image_size = image_size

    def extract_from_image(self, bgr_image):
        bgr_image_r = cv2.resize(bgr_image, self.image_size)
        image = self._change_color_space(bgr_image_r, self.color_space)
        image_features = [x.create(image) for x in self.features_providers]
        return np.concatenate(image_features)

    def extract_from_files(self, image_paths):
        features = []
        for image_path in image_paths:
            bgr_image = cv2.imread(image_path)
            image_features = self.extract_from_image(bgr_image)
            features.append(image_features)

        return np.array(features).astype(np.float64)

    @staticmethod
    def _change_color_space(bgr_image, color_space):
        if color_space is "BGR":
            return np.copy(bgr_image)

        from_color_space = "COLOR_BGR2"

        destination_color_space = from_color_space + color_space
        if hasattr(cv2, destination_color_space):
            destination_color_space_code = getattr(cv2, destination_color_space)
            return cv2.cvtColor(bgr_image, destination_color_space_code)
        else:
            supported_conversion_keys = filter(lambda y: y.startswith(from_color_space), dir(cv2))
            supported_color_spaces = list(map(lambda x: x.replace(from_color_space, ""), supported_conversion_keys))

            msg = "Not supported color space: '{}'. \n\tSupported color spaces: {}.".format(
                color_space,
                ", ".join(supported_color_spaces))
            raise Exception(msg)
