import resnet
from keras.models import model_from_json

def resnet_18(**kwargs):
    input_shape = tuple(kwargs['input_shape'])
    num_outputs = kwargs['num_outputs']
    model = resnet.ResnetBuilder.build_resnet_18(input_shape, num_outputs)

    return model

def resnet_34(**kwargs):
    input_shape = tuple(kwargs['input_shape'])
    num_outputs = kwargs['num_outputs']
    model = resnet.ResnetBuilder.build_resnet_34(input_shape, num_outputs)

    return model

def resnet_50(**kwargs):
    input_shape = tuple(kwargs['input_shape'])
    num_outputs = kwargs['num_outputs']
    model = resnet.ResnetBuilder.build_resnet_50(input_shape, num_outputs)

    return model

def resnet_101(**kwargs):
    input_shape = tuple(kwargs['input_shape'])
    num_outputs = kwargs['num_outputs']
    model = resnet.ResnetBuilder.build_resnet_101(input_shape, num_outputs)

    return model

def resnet_152(**kwargs):
    input_shape = tuple(kwargs['input_shape'])
    num_outputs = kwargs['num_outputs']
    model = resnet.ResnetBuilder.build_resnet_152(input_shape, num_outputs)

    return model

def build_model_from_config(config):
    model_class = globals()[config['class']]
    model_params = config['params']
    model = model_class(**model_params)

    return model

def load_model(json_path, weights_path):
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)

    return model

