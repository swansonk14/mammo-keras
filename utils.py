import os

def create_if_necessary(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_model_paths(model_name):
    paths = dict()
    paths['weights_path'] = 'weights/{}/weights.hdf5'.format(model_name)
    paths['log_dir'] = 'logs/{}'.format(model_name)
    paths['json_path'] = 'model_json/{}.json'.format(model_name)
    paths['results_path'] = 'results/{}.json'.format(model_name)
    paths['viz_dir'] = '/raid/scratch/model_viz/{}'.format(model_name)

    for name, path in paths.iteritems():
        if 'dir' in name:
            dirname = path
        else:
            dirname = os.path.dirname(path)
        create_if_necessary(dirname)
    
    return paths
