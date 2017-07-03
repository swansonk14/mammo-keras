from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.misc import imresize

MEAN = 46.5584534313
STD =  36.8535621221

class Transformer:
	""" The Transformer class is an abstract class which applies a transformation to the provided data. """

	__metaclass__ = ABCMeta

	@abstractmethod
	def transform(self, data):
		pass

class ResizeImage(Transformer):
	def __init__(self, size):
		self.size = size

	def transform(self, image):
		return imresize(image, self.size)

class NormalizeImage(Transformer):
	def __init__(self, mean=MEAN, std=STD):
		self.mean = mean
		self.std = std

	def transform(self, image):
		return (image - self.mean) / self.std

class ReshapeImage(Transformer):
	def __init__(self, shape):
		self.shape = shape

	def transform(self, image):
		return np.reshape(image, self.shape)

class BinarizeLabel(Transformer):
	def transform(self, label):
		return [0, 1] if label == 1 else [1, 0]
