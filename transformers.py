from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.misc import imresize

MEAN = 46.5584534313
STD =  36.8535621221

class Transformer:
    """ The Transformer class is an abstract class which applies a transformation to the provided data. """

    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def transform(self, data):
        pass

class ResizeImage(Transformer):
    def __init__(self, **kwargs):
        size = kwargs['size']
        if type(size) == list:
            size = tuple(size)
        self.size = size

    def transform(self, image):
        return imresize(image, self.size)

class NormalizeImage(Transformer):
    def __init__(self, **kwargs):
        self.mean = float(kwargs.get('mean', MEAN))
        self.std = float(kwargs.get('std', STD))

    def transform(self, image):
        return (image - self.mean) / self.std

class ReshapeImage(Transformer):
    def __init__(self, **kwargs):
        shape = kwargs['shape']
        if type(shape) == list:
            shape = tuple(shape)
        self.shape = shape

    def transform(self, image):
        return np.reshape(image, self.shape)

class BinarizeLabel(Transformer):
    def transform(self, label):
        return [0, 1] if label == 1 else [1, 0]
