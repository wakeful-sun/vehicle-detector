from data_provider import DataProvider
from features_extractor import FeaturesExtractor
from classifier import Classifier
from features.spatial_binning import SpatialBinningOfColorFeaturesFactory
from features.color_histogram import ColorHistogramFeaturesFactory
from features.hog import HistogramOfOrientedGradientsFeaturesFactory


cars_path = "../../training_images/vehicles/vehicles/*/*.png"
non_cars_path = "../../training_images/non-vehicles/non-vehicles/*/*.png"

#   ------
color_space = "HSV"  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
#   ------
spatial_binning_size = (16, 16)
#   ------
color_histogram_n_bins = 16
bins_range = (0, 256)
#   ------
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # possible values are 0, 1, 2 and "ALL"

features_providers = [
    SpatialBinningOfColorFeaturesFactory(size=spatial_binning_size),
    ColorHistogramFeaturesFactory(n_bins=color_histogram_n_bins, bins_range=bins_range),
    HistogramOfOrientedGradientsFeaturesFactory(orient=orient,
                                                pix_per_cell=pix_per_cell,
                                                cell_per_block=cell_per_block,
                                                hog_channel=hog_channel)
]

extractor = FeaturesExtractor(features_providers, color_space)
clf = Classifier(extractor, fit_step=3000)

data = DataProvider(cars_path, non_cars_path, test_size=0.2)
data.save_labels_info_graph()
print("Training is in progress...")
clf.train(data.train.features, data.train.labels)

print("Training time: {:.2f} min.".format(clf.training_time/60))
accuracy = clf.accuracy(data.test.features, data.test.labels)
print("Test accuracy: {:.3f}%.".format(accuracy * 100))

space = "-" * 50
info = [
    "Accuracy: {:.4f}%".format(accuracy * 100),
    space,
    "\t- Features vector compound parts -",
    "features_providers: [{}]".format(", ".join([type(i).__name__ for i in features_providers])),
    "color_space: {}".format(color_space),
    space,
    "\t- Spatial binning of color parameters -",
    "spatial_binning_size: {}".format(spatial_binning_size),
    space,
    "\t- Color histogram parameters -",
    "color_histogram_n_bins: {}".format(color_histogram_n_bins),
    "bins_range: {}".format(bins_range),
    space,
    "\t- Histogram of oriented gradients parameters -",
    "orient: {}".format(orient),
    "pix_per_cell: {}".format(pix_per_cell),
    "cell_per_block: {}".format(cell_per_block),
    "hog_channel: {}".format(hog_channel),
    space
]
info.extend(clf.summary)
info.append(space)
info.append(space)
info.extend(data.summary)


def save_summary(path, acc, content):
    with open("{}_summary_{:.3f}.txt".format(path, acc * 100), "w") as f:
        f.write("\n".join(content))


clf.save_model("../training_results/model.pkl")
clf.save_scaler("../training_results/scaler.pkl")
save_summary("../training_results/", accuracy, info)
