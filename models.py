import os
import resnet
from keras.models import model_from_json

def resnet_18(image_size, channel_size, num_outputs):
    input_shape = (channel_size, image_size, image_size)
    model = resnet.ResnetBuilder.build_resnet_18(input_shape, num_outputs)

    return model

def resnet_34(image_size, channel_size, num_outputs):
    input_shape = (channel_size, image_size, image_size)
    model = resnet.ResnetBuilder.build_resnet_34(input_shape, num_outputs)

    return model

def resnet_50(image_size, channel_size, num_outputs):
    input_shape = (channel_size, image_size, image_size)
    model = resnet.ResnetBuilder.build_resnet_50(input_shape, num_outputs)

    return model

def resnet_101(image_size, channel_size, num_outputs):
    input_shape = (channel_size, image_size, image_size)
    model = resnet.ResnetBuilder.build_resnet_101(input_shape, num_outputs)

    return model

def resnet_152(image_size, channel_size, num_outputs):
    input_shape = (channel_size, image_size, image_size)
    model = resnet.ResnetBuilder.build_resnet_152(input_shape, num_outputs)

    return model

def build_model(model_architecture, image_size, channel_size, num_outputs):
    model_class = globals()[model_architecture]
    model = model_class(image_size, channel_size, num_outputs)

    return model

def load_model(json_path, weights_path):
    if not os.path.exists(json_path):
        print('Can\'t load model: json path does not exist')
        return None
    if not os.path.exists(weights_path):
        print('Can\'t load model: weights path does not exist')
        return None

    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)

    return model

