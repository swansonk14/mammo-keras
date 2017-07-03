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
		key_array = kwargs['key_array']
		# If key_array is not a list, convert it to a list
		if type(key_array) != list:
			self.key_array = [key_array]
		else:
			self.key_array = key_array

	# Returns `row[key_array[0]][key_array[1]]...`
	def extract(self, row):
		data = row
		for key in self.key_array:
			data = data[key]

		return data

