from data_provider import DataProvider
from features_extractor import FeaturesExtractor
from classifier import Classifier
from features.spatial_binning import SpatialBinningOfColorFeaturesFactory
from features.color_histogram import ColorHistogramFeaturesFactory
from features.hog import HistogramOfOrientedGradientsFeaturesFactory
from detector import Detector


cars_path = r"..\03_training_images\vehicles_smallset\vehicles_smallset\cars1\*.jpeg"
non_cars_path = r"..\03_training_images\non-vehicles_smallset\non-vehicles_smallset\notcars1\*jpeg"
test_image_path = r"..\01_images\bbox-example-image.jpg"

#   ------
color_space = "HSV"  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
#   ------
spatial_binning_size = (16, 16)
#   ------
color_histogram_n_bins = 16
#   ------
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # possible values are 0, 1, 2 and "ALL"

features_providers = [
    SpatialBinningOfColorFeaturesFactory(size=spatial_binning_size),
    ColorHistogramFeaturesFactory(n_bins=color_histogram_n_bins, bins_range=(0, 256)),
    HistogramOfOrientedGradientsFeaturesFactory(orient=orient,
                                                pix_per_cell=pix_per_cell,
                                                cell_per_block=cell_per_block,
                                                hog_channel=hog_channel)
]

extractor = FeaturesExtractor(features_providers, color_space)


data = DataProvider(extractor, cars_path, non_cars_path, test_size=0.2)
print('Feature vector length:', len(data.train.features[0]))

clf = Classifier()
clf.train(data.train.features, data.train.labels)

print(round(clf.training_time, 2), 'Seconds to train SVC...')
print('Test Accuracy of SVC = ', clf.accuracy(data.test.features, data.test.labels))

#TODO: visualize
#TODO: save the result