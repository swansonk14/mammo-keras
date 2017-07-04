import argparse
import json
import yaml
import os
from models import load_model, build_model
from pipeline import build_pipeline_from_config
from generators import batch_generator
from evaluation import evaluate_performance_on_groups
from utils import get_model_paths
import resnet
import keras
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(description='Flags for model')
parser.add_argument('--data_path', type=str, required=True, help='path to metadata')
parser.add_argument('--model_architecture', type=str, required=True, help='network architecture to use')
parser.add_argument('--model_name', type=str, required=True, help='name of the model')
parser.add_argument('--image_pipeline', type=str, required=True, help='name of the yaml file with the config for the image pipeline (must be in "pipelines/image")')
parser.add_argument('--label_pipeline', type=str, required=True, help='name of the yaml file with the config for the label pipeline (must be in "pipelines/label"')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--examples_per_epoch', type=int, default=20000, help='number of examples to train on in each epoch')
parser.add_argument('--examples_per_val', type=int, default=5000, help='number of examples to validate on after each epoch (determines which sets of parameters are saved)')
parser.add_argument('--examples_per_eval', type=int, default=2000, help='number of examples to evaluate on after each epoch')
parser.add_argument('--load_model', action='store_true', default=False, help='True to load the model from the previous best chckpoint')
parser.add_argument('--debug', action='store_true', default=False, help='True to print exceptions')
args = parser.parse_args()

paths = get_model_paths(args.model_name)

print('Loading configs...')
with open('pipelines/image/{}.yaml'.format(args.image_pipeline), 'r') as config_file:
    image_pipeline_config = yaml.load(config_file)

image_size = image_pipeline_config['image_size']
channel_size = image_pipeline_config['channel_size']

with open('pipelines/label/{}.yaml'.format(args.label_pipeline), 'r') as config_file:
    label_pipeline_config = yaml.load(config_file)

num_outputs = label_pipeline_config['num_outputs']

model = None
if args.load_model:
    print('Loading model from checkpoint...')
    model = load_model(paths['json_path'], paths['weights_path'])
if model is None:
    print('Building model...')
    model = build_model(args.model_architecture, image_size, channel_size, num_outputs)
print(model.summary())

optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

print('Building data pipelines...')
image_pipeline = build_pipeline_from_config(image_pipeline_config)
label_pipeline = build_pipeline_from_config(label_pipeline_config)

print('Loading metadata...')
with open(args.data_path, 'r') as data_file:
    metadata = json.load(data_file)

train_metadata = [row for row in metadata if row.get('split_group', None) == 'train']
dev_metadata = [row for row in metadata if row.get('split_group', None) == 'dev']
test_metadata = [row for row in metadata if row.get('split_group', None) == 'test']

print('Metadata length = {}'.format(len(metadata)))
print('Train metadata length = {}'.format(len(train_metadata)))
print('Dev metadata length = {}'.format(len(dev_metadata)))
print('Test metadata length = {}'.format(len(test_metadata)))

print('Creating generators...')
train_generator = batch_generator(train_metadata, batch_size=args.batch_size, image_pipeline=image_pipeline, label_pipeline=label_pipeline, debug=args.debug)
dev_generator = batch_generator(dev_metadata, batch_size=args.batch_size, image_pipeline=image_pipeline, label_pipeline=label_pipeline, debug=args.debug)
test_generator = batch_generator(test_metadata, batch_size=args.batch_size, image_pipeline=image_pipeline, label_pipeline=label_pipeline, debug=args.debug)

group_generators = [(train_generator, 'train'), (dev_generator, 'dev'), (test_generator, 'test')]

checkpoint = ModelCheckpoint(paths['weights_path'], monitor='val_loss', verbose=1, save_best_only=True, mode='min')

with open(paths['json_path'], 'w') as json_file:
    json_file.write(model.to_json())

class Histories(keras.callbacks.Callback):
    def __init__(self, args):
        super(Histories, self).__init__()
        self.results = {'flags': vars(args), 'results': []}

    def on_epoch_end(self, epoch, logs={}):
        epoch_results = evaluate_performance_on_groups(model, group_generators, steps_per_eval=int(args.examples_per_eval / args.batch_size))
        epoch_results['epoch'] = epoch
        self.results['results'].append(epoch_results)
        with open(paths['results_path'], 'w') as results_file:
            json.dump(self.results, results_file, indent=4, sort_keys=True)

print('Begin training...')
model.fit_generator(train_generator,
                    steps_per_epoch=int(args.examples_per_epoch / args.batch_size),
                    verbose=1,
                    epochs=1000,
                    validation_data=dev_generator,
                    validation_steps=int(args.examples_per_val / args.batch_size),
                    callbacks=[TensorBoard(log_dir=paths['log_dir']), checkpoint, Histories(args)])
