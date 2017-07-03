from abc import ABCMeta, abstractmethod

class Extractor:
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def extract(self, row):
        pass

class Identity(Extractor):
    def extract(self, row):
        return row

class Key(Extractor):
    """ Extracts a piece of information from a dictionary using a list of keys. """

    def __init__(self, **kwargs):
        keys = kwargs['keys']
        if type(keys) != list:
            keys = [keys]
        self.keys = keys

    # Returns `row[keys[0]][keys[1]]...`
    def extract(self, row):
        data = row
        for key in self.keys:
            data = data[key]

        return data

