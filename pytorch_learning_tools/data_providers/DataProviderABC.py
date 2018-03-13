from abc import ABC, abstractmethod


class DataProviderABC(ABC):

    # which inds belong to which splits?
    @property
    @abstractmethod
    def splits(self):
        pass

    # get a single data point using unique id - not how you want to iterate over data but useful for inspection
    @abstractmethod
    def __getitem__(self, unique_id):
        pass
