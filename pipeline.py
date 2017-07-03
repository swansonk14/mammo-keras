import extractors
import loaders

class Pipeline:
	def __init__(self, extractor=extractors.Identity(), loader=loaders.Identity(), transformers=[]):
		self.extractor = extractor
		self.loader = loader
		self.transformers = transformers

	def process(self, row):
		data = self.extractor.extract(row)
		data = self.loader.load(data)
		for transformer in self.transformers:
			data = transformer.transform(data)

		return data
