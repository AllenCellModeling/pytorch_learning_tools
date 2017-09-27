from abc import ABC, abstractmethod


class DataProviderABC(ABC):

    @abstractmethod
    def get_len(self, split):
        pass

    @abstractmethod
    def get_unique_targets(self):
        pass

    @abstractmethod
    def get_data_paths(self, inds, split):
        pass

    @abstractmethod
    def get_random_sample(self, N, split):
        pass

    @abstractmethod
    def get_data_points(self, inds, split):
        pass
