from abc import ABC, abstractmethod


class BaseEdgeMeasure(ABC):
    def __init__(self, network):
        self.network = network

    @abstractmethod
    def measure(self):
        pass

    def print_info(self) -> None:
        measures = self.measure()
        sorted_measures = sorted(measures.items(), key=lambda item: item[1], reverse=True)
        print(f"{self.__class__.__name__} (Highest to Lowest):")
        for edge, value in sorted_measures:
            print(f"Edge {edge}: {value}")