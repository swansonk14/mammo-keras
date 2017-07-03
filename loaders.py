from abc import ABCMeta, abstractmethod
from scipy.misc import imread

class Loader:
	""" The Loader class is an abstract class which loads data from storage. """

	__metaclass__ = ABCMeta

	@abstractmethod
	def load(self, input):
		pass

class Identity(Loader):
	def load(self, input):
		return input

class GrayscaleImage(Loader):
	def load(self, path):
		return imread(path, mode='L')
