from abc import ABC, abstractmethod


class BaseNodeMeasure(ABC):
    def __init__(self, network):
        self.network = network

    def measure(self):
        pass
