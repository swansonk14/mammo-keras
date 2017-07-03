import argparse
import json
import extractors
import loaders
import transformers
from pipeline import Pipeline
from generators import batch_generator
from evaluation import evaluate_performance_on_groups
from utils import get_model_paths
import resnet
import keras
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(description='Flags for model')
parser.add_argument('--data_path', type=str, required=True, help='path to metadata')
parser.add_argument('--model_name', type=str, required=True, help='name of the model')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--image_size', type=int, default=256, help='image size')
parser.add_argument('--channel_size', type=int, default=1, help='number of image channels')
parser.add_argument('--label_size', type=int, default=2, help='size of label vector')
parser.add_argument('--path_keys', type=str, default='image_path_256', help='keys of image path in metadata')
parser.add_argument('--label_keys', type=str, default='density', help='keys of label in metadata')
parser.add_argument('--examples_per_epoch', type=int, default=1000, help='number of examples to train on in each epoch')
parser.add_argument('--examples_per_val', type=int, default=2000, help='number of examples to validate on after each epoch (determines which sets of parameters are saved)')
parser.add_argument('--examples_per_eval', type=int, default=1000, help='number of examples to evaluate on after each epoch')
parser.add_argument('--debug', action='store_true', help='True to print exceptions')
args = parser.parse_args()

print('Loading model...')
model = resnet.ResnetBuilder.build_resnet_18((args.channel_size, args.image_size, args.image_size), args.label_size)
print(model.summary())

optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

paths = get_model_paths(args.model_name)

print('Building data pipelines...')
image_extractor = extractors.Key(args.path_keys)
image_loader = loaders.GrayscaleImage()
image_transformers = [transformers.ResizeImage((args.image_size, args.image_size)),
					  transformers.NormalizeImage(),
					  transformers.ReshapeImage((args.image_size, args.image_size, args.channel_size))]
image_pipeline = Pipeline(image_extractor, image_loader, image_transformers)

label_extractor = extractors.Key(args.label_keys)
label_loader = loaders.Identity()
label_transformers = [transformers.BinarizeLabel()]
label_pipeline = Pipeline(label_extractor, label_loader, label_transformers)

print('Loading metadata...')
with open(args.data_path, 'r') as data_file:
	metadata = json.load(data_file)
print('Metadata length = {}'.format(len(metadata)))

print('Creating generators...')
train_generator = batch_generator('train', metadata, batch_size=args.batch_size, image_pipeline=image_pipeline, label_pipeline=label_pipeline, debug=args.debug)
dev_generator = batch_generator('dev', metadata, batch_size=args.batch_size, image_pipeline=image_pipeline, label_pipeline=label_pipeline, debug=args.debug)
test_generator = batch_generator('test', metadata, batch_size=args.batch_size, image_pipeline=image_pipeline, label_pipeline=label_pipeline, debug=args.debug)

group_generators = [(train_generator, 'train'), (dev_generator, 'dev'), (test_generator, 'test')]

checkpoint = ModelCheckpoint(paths['checkpoint_path'], monitor='val_loss', verbose=1, save_best_only=True, mode='min')

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
