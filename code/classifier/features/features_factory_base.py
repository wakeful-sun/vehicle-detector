from abc import ABC, abstractmethod


class FeaturesFactory(ABC):

    @abstractmethod
    def create(self, image):
        raise NotImplementedError("Should have implemented this")
