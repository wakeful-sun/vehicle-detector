import glob
import numpy as np
from sklearn.model_selection import train_test_split
import random


class DataProvider:

    def __init__(self, cars_path, non_cars_path, test_size=0.2, limit=None):
        non_car_class_id = 0
        self.car_class_id = 1

        cars_paths = glob.glob(cars_path)
        random.shuffle(cars_paths)
        cars_paths = cars_paths[:limit]
        non_cars_paths = glob.glob(non_cars_path)
        random.shuffle(non_cars_paths)
        non_cars_paths = non_cars_paths[:limit]

        features = np.concatenate((cars_paths, non_cars_paths))
        labels = np.concatenate((np.full(len(cars_paths), self.car_class_id),
                                 np.full(len(non_cars_paths), non_car_class_id)))

        rand_state = np.random.randint(0, 100)
        x_train, x_test, y_train, y_test = train_test_split(features, labels,
                                                            test_size=test_size, random_state=rand_state)

        self.train_set = DataSet(x_train, y_train)
        self.test_set = DataSet(x_test, y_test)

    @property
    def train(self):
        return self.train_set

    @property
    def test(self):
        return self.test_set

    @property
    def car_class(self):
        return self.car_class_id
    #
    # def get_features_normalized(self, rgb_image):
    #     features = self.features_extractor.extract_from_image(rgb_image)
    #     return self.scaler.transform(np.array(features).reshape(1, -1))


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
