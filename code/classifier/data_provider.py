import numpy as np
from os import path
import cv2
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tools


class DataProvider:

    def __init__(self, cars_paths, non_cars_paths, car_id=1, non_car_id=0, test_size=0.2):
        non_car_class_id = non_car_id
        self.car_class_id = car_id

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

        space = "***"
        info = [
            "\tData set information:",
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
            space
        ]

        def get_random_images_summary(paths, n):
            indexes = random.sample(range(len(paths)), n)
            f_summary = []
            for i in indexes:
                file_path = paths[i]
                f_name, f_ext =path.splitext(file_path)
                image = cv2.imread(file_path)
                f_summary.append("Name: {}, Extension: {}, Shape: {}".format(f_name, f_ext, image.shape))
            return f_summary

        info.append("Some train images info:")
        info.extend(get_random_images_summary(self.train_set.features, 6))

        info.append("Some test images info:")
        info.extend(get_random_images_summary(self.test_set.features, 6))

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

    def save_random_images(self, class_id, path):

        labels, features = np.array(self.train_set.labels), self.train_set.features
        class_indexes = np.where(labels == class_id)[0]

        def get_random_images_of_class(paths, c_indexes, n):
            indexes = random.sample(range(len(c_indexes)), n)
            rgb_images = []
            for i in indexes:
                file_path = paths[c_indexes[i]]
                image = cv2.imread(file_path)
                rgb_images.append(image)

            return rgb_images

        images = get_random_images_of_class(features, class_indexes, 20)
        composite_image = tools.create_composite_image(images, h_span=5, v_span=5, n_columns=5)

        cv2.imwrite(path, composite_image)


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
