import transformers

def build_pipeline_from_config(config):
    transformer_array = []
    for transformer_config in config.get('transformers', []):
        transformer_class = getattr(transformers, transformer_config['class'])
        transformer_params = transformer_config.get('params', dict())
        transformer = transformer_class(**transformer_params)
        transformer_array.append(transformer)

    return Pipeline(transformer_array)

class Pipeline:
    def __init__(self, transformers=[]):
        self.transformers = transformers

    def process(self, row):
        for transformer in self.transformers:
            data = transformer.transform(data)

        return data
