import process_results
import argparse
from pprint import pprint

def main(filename, min_epoch, max_epoch, max_group, max_measure, reverse):
    results = process_results.get_results(filename)

    if not min_epoch:
        min_epoch = 0

    if max_epoch != 0:
        max_epoch = max_epoch if max_epoch else results[-1]['epoch']
    max_group = max_group if max_group else 'test'
    max_measure = max_measure if max_measure else 'f1'

    max_statistic, max_results = process_results.get_max_statistic(results, min_epoch, max_epoch, max_group, max_measure, reverse)
    
    max_min = 'Minimum' if reverse else 'Maximum'

    print('')
    print('{} {} {} of {:.3f} on epoch {:d} in epoch range {:d} to {:d}'.format(max_min, max_group, max_measure, max_results[max_group][max_measure], max_results['epoch'], min_epoch, max_epoch))
    print('')
    print('epoch {}'.format(max_results['epoch']))
    print('')
    print('train results')
    pprint(max_results['train'])
    print('')
    print('dev results')
    pprint(max_results['dev'])
    print('')
    print('test results')
    pprint(max_results['test'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help='json file with results', required=True)
    parser.add_argument('--min_epoch', help='epoch to start saerch; default is 0', type=int)
    parser.add_argument('--max_epoch', help='epoch to end search; default is last epoch', type=int)
    parser.add_argument('--group', help='train, dev, test, or grid; default is test')
    parser.add_argument('--measure', help='measure to find maximum of; default is f1')
    parser.add_argument('--reverse', action='store_true', default=False, help='True to compute min instead of max')
    args = parser.parse_args()

    main(args.filename, args.min_epoch, args.max_epoch, args.group, args.measure, args.reverse)
