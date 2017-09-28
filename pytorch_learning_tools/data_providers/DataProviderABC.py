from abc import ABC, abstractmethod


class DataProviderABC(ABC):

    @abstractmethod
    def data_len(self, split):
        pass

    @abstractmethod
    def target_cardinality(self):
        pass

    @abstractmethod
    def get_data_points(self, inds, split):
        pass
