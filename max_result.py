import sys
import numpy as np
import process_results
import argparse

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

    print
    print '{} {} {} of {:.3f} on epoch {:d} in epoch range {:d} to {:d}'.format(max_min, max_group, max_measure, max_results[max_group][max_measure], max_results['epoch'], min_epoch, max_epoch)
    print  '{} {} {} of {:.3f} on epoch {:d} in epoch range {:d} to {:d}'.format(max_min, 'test', max_measure, max_results['test'][max_measure], max_results['epoch'], min_epoch, max_epoch)
    print

    for group in ['train', 'dev', 'test', 'grid']:
        try:
            #print '{} accuracy {:.3f}%'.format(group, 100*max_results[group]['accuracy'])
            print '{} auc {}'.format(group, max_results[group]['auc'])
            print '{} f1 {}'.format(group, max_results[group]['f1'])  
            print '{} precision {}'.format(group, max_results[group]['precision'])  
            print '{} recall {}'.format(group, max_results[group]['recall'])  
            print '{} confusion matrix'.format(group)
            print np.array(max_results[group]['confusion_matrix'])
            print
        except Exception as e:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help='json file with results', required=True)
    parser.add_argument('--min_epoch', help='epoch to start saerch; default is 0', type=int)
    parser.add_argument('--max_epoch', help='epoch to end search; default is last epoch', type=int)
    parser.add_argument('--group', help='train, dev, test, or grid; default is test')
    parser.add_argument('--measure', help='measure to find maximum of; default is f1')
    parser.add_argument('--reverse', help='True to compute min instead of max', type=str, default='False')
    args = parser.parse_args()

    reverse = args.reverse == 'True'

    main(args.filename, args.min_epoch, args.max_epoch, args.group, args.measure, reverse)
