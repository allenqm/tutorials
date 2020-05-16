"""Re-useable components for an ML pipeline"""
from abc import ABC, abstractclassmethod


class DataBlock(ABC):

    def __init__(self, datablock=None):
        if datablock:
            self.cached_data = datablock.cached_data
        else:
            self.cached_data = {}
    @abstractclassmethod
    def fetch_data(self):
        """get raw data out of datastore and into in-memory data structure"""
        pass
    @abstractclassmethod
    def preprocess_data(self):
        """return a dataframe with the appropriate shape, names, and dtypes"""
        pass
    @abstractclassmethod
    def split_data(self):
        """return a dict of dataframes with the appropriate shape, names, and dtypes"""
        pass

class Learner(ABC):

    def __init__(self, DataBlock):
        """network definition.  initialization of a model class"""
        pass

    @abstractclassmethod
    def transform(self):
        """transform to algo-specific data containers.  additional transformations that we don't want to leak across folds"""
        pass

    @abstractclassmethod
    def fit(self):
        """The training loop"""
        pass

    @abstractclassmethod
    def get_output(self):
        """prediction or 'inference'"""
        pass


def evaluate(Learner, DataBlock):
    pass

def compose(dataframe, list_of_funcs):
    pass



'''
db2 = ImgDataBlock(db1) # this preserves the cached_data of before, but make use of the new methods
db2.preprocess_data()
'''

class ImageDataBlock(DataBlock):

    def fetch_data(self):
        pass

    def preprocess_data(self):
        pass

    def split_data(self):
        pass