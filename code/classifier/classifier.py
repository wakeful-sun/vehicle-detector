import time
from sklearn.svm import LinearSVC


class Classifier:

    def __init__(self):
        self.classifier = None
        self.t_time = 0

    @property
    def training_time(self):
        return self.t_time

    def train(self, train_features, train_labels):
        t_start = time.time()
        self.classifier = LinearSVC()
        self.classifier.fit(train_features, train_labels)
        t_end = time.time()
        self.t_time = t_end - t_start

    def accuracy(self, test_features, test_labels):
        return round(self.classifier.score(test_features, test_labels), 4)

    def predict(self, feature):
        return self.classifier.predict(feature)
