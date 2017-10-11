from abc import ABC, abstractmethod


class DataProviderABC(ABC):

    # which inds belong to which splits?
    @property
    @abstractmethod
    def splits(self):
        pass
    
    # hoq many classes are we predicting?
    @property
    @abstractmethod
    def classes(self):
        pass

    # get data using unique ids
    # not how yoyu want to iterate over data but useful for inspection
    @abstractmethod
    def get_data_points(self, unique_ids):
        pass
