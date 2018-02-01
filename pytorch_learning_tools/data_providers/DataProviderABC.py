from abc import ABC, abstractmethod


class DataProviderABC(ABC):

    # Which inds belong to which splits?
    @property
    @abstractmethod
    def splits(self):
        pass
    
    # How many classes are we predicting?
    @property
    @abstractmethod
    def classes(self):
        pass

    # Get data using unique ids
    # Not how you want to iterate over data but useful for inspection
    @abstractmethod
    def get_data_points(self, unique_ids):
        pass
