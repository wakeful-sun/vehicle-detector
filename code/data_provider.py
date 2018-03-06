import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataProvider:

    def __init__(self, features_extractor, cars_path, non_cars_path, test_size=0.2):
        self.car_class_id = 1
        self.non_car_class_id = 0

        cars_paths = glob.glob(cars_path)
        non_cars_paths = glob.glob(non_cars_path)

        car_features = features_extractor.extract_from_files(cars_paths)
        non_car_features = features_extractor.extract_from_files(non_cars_paths)

        features = np.vstack((car_features, non_car_features)).astype(np.float64)
        labels = np.hstack((np.full(len(car_features), self.car_class_id),
                            np.full(len(non_car_features), self.non_car_class_id)))

        self.scaler, (x_train, y_train, x_test, y_test) = self._normalize_and_split(features, labels, test_size)

        self.train_set = DataSet(x_train, y_train)
        self.test_set = DataSet(x_test, y_test)
        self.features_extractor = features_extractor

    @staticmethod
    def _normalize_and_split(features, labels, test_size):
        rand_state = np.random.randint(0, 100)
        x_train, x_test, y_train, y_test = train_test_split(features, labels,
                                                            test_size=test_size, random_state=rand_state)

        x_scaler = StandardScaler().fit(x_train)
        x_train = x_scaler.transform(x_train)
        x_test = x_scaler.transform(x_test)

        return x_scaler, (x_train, y_train, x_test, y_test)

    @property
    def train(self):
        return self.train_set

    @property
    def test(self):
        return self.test_set

    @property
    def car_class(self):
        return self.car_class_id

    @property
    def non_car_class(self):
        return self.non_car_class_id

    def get_features_normalized(self, rgb_image):
        features = self.features_extractor.extract_from_image(rgb_image)
        return self.scaler.transform(np.array(features).reshape(1, -1))


class DataSet:

    def __init__(self, features, labels):
        self.f = features
        self.l = labels

    @property
    def features(self):
        return self.f

    @property
    def labels(self):
        return self.l
