from features.features_factory_base import FeaturesFactory
from skimage.feature import hog


class HistogramOfOrientedGradientsFeaturesFactory(FeaturesFactory):
    
    def __init__(self, orient, pix_per_cell, cell_per_block, hog_channel="ALL", feature_vec=True, visualise=False):

        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.feature_vec = feature_vec
        self.visualise = visualise

        if self.visualise:
            raise NotImplementedError("Visualisation is not implemented yet")

    def create(self, image):

        hog_features = []
        if self.hog_channel == "ALL":
            for channel in range(image.shape[2]):
                channel_features, _ = self._extract_channel_features(image, channel)
                hog_features.extend(channel_features)
        else:
            hog_features, _ = self._extract_channel_features(image, self.hog_channel)

        return hog_features

    def _extract_channel_features(self, image, channel):
        assert type(channel) is int and 0 <= channel <= image.shape[2], "invalid color channel {}.".format(channel)
        pixels_per_cell = (self.pix_per_cell, self.pix_per_cell)
        cells_per_block = (self.cell_per_block, self.cell_per_block)

        result = hog(image[:, :, channel], orientations=self.orient,
                     pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                     block_norm="L2-Hys", visualise=self.visualise, transform_sqrt=False,
                     feature_vector=self.feature_vec)

        if self.visualise:
            features_vector, hog_image = result
        else:
            features_vector, hog_image = result, None

        return features_vector, hog_image