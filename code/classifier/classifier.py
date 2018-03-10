import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import numpy as np


class Classifier:

    def __init__(self, features_extractor, fit_step=500):
        self.features_extractor = features_extractor
        self.fit_step = fit_step

        self.clf = None
        self.scaler = None

        self.t_time = 0
        self.n_training = 0
        self.n_testing = 0

    @property
    def training_time(self):
        return self.t_time

    @property
    def summary(self):
        return [
            "Training items count: {}".format(self.n_training),
            "Testing items count: {}".format(self.n_testing),
            "Training time: {:.2f} minutes".format(self.t_time/60)
        ]

    def train(self, train_feature_paths, train_labels):
        dataSet = DataSet(train_feature_paths, train_labels)
        self.clf = LinearSVC()
        self.scaler = StandardScaler()

        # Can be replaced with while loop for classifier that support partial fit
        # LinearSVC does not support partial fit
        if dataSet.move_next(self.fit_step):
            t_start = time.time()
            step_features, step_labels = dataSet.current

            features = self.features_extractor.extract_from_files(step_features)
            self.scaler.fit(features)
            features_norm = self.scaler.transform(features)
            self.clf.fit(features_norm, step_labels)

            self.n_training = self.n_training + len(features)
            self.t_time = self.t_time + time.time() - t_start
        else:
            raise Exception("no training data provided")

    def accuracy(self, test_feature_paths, test_labels):
        features = self.features_extractor.extract_from_files(test_feature_paths)
        self.n_testing = len(features)
        return self.clf.score(features, test_labels)

    def predict(self, rgb_image):
        features = self.features_extractor.extract_from_image(rgb_image)
        features_norm = self.scaler.transform(np.array(features).reshape(1, -1))
        return self.clf.predict(features_norm)

    def save_model(self, path):
        joblib.dump(self, path)


class DataSet:

    def __init__(self, features, labels):
        self.f_len = len(features)
        assert self.f_len == len(labels), "invalid features/labels set"

        self.f = features
        self.l = labels
        self.start = 0
        self.current_batch = None

    def _range(self, items, batch_size):
        if batch_size is None:
            return items[self.start:]
        return items[self.start:self.start + batch_size]

    def move_next(self, batch_size=None):
        self.current_batch = self._range(self.f, batch_size), self._range(self.l, batch_size)
        if len(self.current_batch[0]):
            self.start = self.start + batch_size
            return True
        else:
            self.reset()
            return False

    def reset(self):
        self.start = 0
        self.current_batch = None

    @property
    def current(self):
        return self.current_batch
