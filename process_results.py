import json

def get_data(filename):
    with open(filename, 'r') as data_file:
        return json.load(data_file)

def get_results(filename):
    data = get_data(filename)
    return data['results']

def get_flags(filename):
    data = get_data(filename)
    return data['flags']

def get_max_statistic(results, min_epoch, max_epoch, group, measure, reverse):
    if max_epoch < 0:
        print 'Invalid epoch: ', max_epoch
        exit()
    if min_epoch > max_epoch:
        print 'Invalid epoch range; {:d} to {:d}'.format(min_epoch, max_epoch)
        exit()
    if group not in ['train', 'dev', 'test', 'grid']:
        print 'Invalid group: ', group
        exit()
    if measure not in results[0]['train'].keys():
        print 'Invalid measure: ', measure
        exit()
    max_score = -float('inf')
    max_score_results = {}
    for i in range(min_epoch, max_epoch+1):
        result = results[i]
        score = result[group][measure]
        if reverse:
            score *= -1
        if score > max_score:
            max_score = score
            max_score_results = result
    if reverse:
        max_score *= -1
    return max_score, max_score_results

