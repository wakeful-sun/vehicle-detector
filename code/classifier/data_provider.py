import glob
import numpy as np
from os import path
import cv2
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DataProvider:

    def __init__(self, cars_path, non_cars_path, test_size=0.2):
        non_car_class_id = 0
        self.car_class_id = 1

        cars_paths = glob.glob(cars_path)
        non_cars_paths = glob.glob(non_cars_path)

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

    @property
    def summary(self):
        train_classes = set(self.train_set.labels)
        test_classes = set(self.test_set.labels)

        def get_random_files_summary(paths, n):
            indexes = random.sample(range(len(paths)), n)
            f_summary = []
            for i in indexes:
                file_path = paths[i]
                f_name, f_ext =path.splitext(file_path)
                image = cv2.imread(file_path)
                f_summary.append("Name: {}, Extension: {}, Shape: {}".format(f_name, f_ext, image.shape))
            return f_summary

        train_images_info = get_random_files_summary(self.train_set.features, 5)
        test_images_info = get_random_files_summary(self.test_set.features, 5)

        space = "***"
        info = [
            "\tData information:",
            space,
            "Car class ID: {}".format(self.car_class_id),
            space,
            "Training items amount: {}".format(len(self.train_set.features)),
            "Car images amount: {}".format(sum(1 for x in self.train_set.labels if x == self.car_class_id)),
            "Training classes: {}".format(train_classes),
            "Training classes amount: {}".format(len(train_classes)),
            space,
            "Testing items amount: {}".format(len(self.test_set.features)),
            "Testing classes: {}".format(test_classes),
            "Testing classes amount: {}".format(len(test_classes)),
            space,
            "Some train images info:"
        ]
        info.extend(train_images_info)
        info.append("Some train images info:")
        info.extend(test_images_info)

        return info

    def save_labels_info_graph(self):
        def save_labels_info(labels, path, name=""):
            fig, ax = plt.subplots()
            fig.suptitle(name, fontweight="bold")

            ax.hist(labels, facecolor="green", alpha=0.75)
            ax.xaxis.grid(which="both")
            ax.set_xticks(list(set(labels)))

            plt.xlabel("classes")
            plt.ylabel("amount of samples")
            plt.savefig(path)

        def save_labels_distribution(labels, path):
            a = range(len(labels))
            fig, ax = plt.subplots()
            fig.suptitle("Labels distribution", fontweight="bold")

            ax.plot(labels, a, "b.", alpha=0.75)
            ax.xaxis.grid(which="both")
            ax.set_xticks(list(set(labels)))

            plt.ylabel("index")
            plt.xlabel("class")
            plt.savefig(path)

        save_labels_info(self.train_set.labels, "../training_results/train_labels.png", "Train set")
        save_labels_info(self.test_set.labels, "../training_results/test_labels.png", "Test set")
        save_labels_distribution(self.train_set.labels, "../training_results/labels_distribution.png")

    def save_visualization(self):
        pass
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
