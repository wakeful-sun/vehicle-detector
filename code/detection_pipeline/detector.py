import cv2


class Detector:

    def __init__(self, data_provider, classifier, sliding_windows_factory):
        self.data_provider = data_provider
        self.classifier = classifier
        self.sliding_windows_factory = sliding_windows_factory
        self.color = (0, 0, 255)
        self.thick = 6

    def _get_hot_windows(self, rgb_image):
        sliding_windows = self.sliding_windows_factory.create(rgb_image.shape)

        on_windows = []
        for window in sliding_windows:
            test_img = cv2.resize(rgb_image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            test_features = self.data_provider.get_features_normalized(test_img)
            prediction_result = self.classifier.predict(test_features)
            if prediction_result == self.data_provider.car_class:
                on_windows.append(window)
        return on_windows

    def detect(self, rgb_image):
        hot_windows = self._get_hot_windows(rgb_image)
        for car_window in hot_windows:
            p1, p2 = car_window
            cv2.rectangle(rgb_image, p1, p2, self.color, self.thick)
