import extractors
import loaders
import transformers

def create_pipeline_from_config(config):
	if 'extractor' in config:
		extractor_class = getattr(extractors, config['extractor']['class'])
		extractor_params = config['extractor'].get('params', dict())
		extractor = extractor_class(**extractor_params)
	else:
		extractor = extractors.Identity()

	if 'loader' in config:
		loader_class = getattr(loaders, config['loader']['class'])
		loader_params = config['loader'].get('params', dict())
		loader = loader_class(**loader_params)
	else:
		loader = loaders.Identity()

	transformer_array = []
	for transformer_config in config.get('transformers', []):
		transformer_class = getattr(transformers, transformer_config['class'])
		transformer_params = transformer_config.get('params', dict())
		transformer = transformer_class(**transformer_params)
		transformer_array.append(transformer)

	return Pipeline(extractor, loader, transformer_array)

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
