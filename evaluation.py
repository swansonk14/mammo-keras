from tqdm import tqdm
from pprint import pprint
import numpy as np
from sklearn import metrics

def compute_results(labels, preds):
    results = dict()
    results['auc'] = metrics.roc_auc_score(labels[:, 1], preds[:, 1])
    fpr, tpr, thresholds = metrics.roc_curve(labels[:, 1], preds[:, 1], drop_intermediate=False)
    results['fpr'], results['tpr'], results['thresholds'] = fpr.tolist(), tpr.tolist(), thresholds.tolist()

    # Convert probabilities to binary
    labels = np.argmax(labels, axis=1)
    preds = np.argmax(preds, axis=1)
    results = dict()
    results['precision'] = metrics.precision_score(labels, preds)
    results['recall'] = metrics.recall_score(labels, preds)
    results['f1'] = metrics.f1_score(labels, preds)
    results['confusion_matrix'] = metrics.confusion_matrix(labels, preds).tolist()
    results['accuracy'] = metrics.accuracy_score(labels, preds)

    return results

def evaluate_performance_on_groups(model, group_generators, steps_per_eval=None, verbose=True):
    results = dict()
    for generator, group in group_generators:
        all_labels = []
        all_preds = []

        for i in tqdm(range(steps_per_eval)):
            images, labels = generator.next()
            preds = model.predict(images)
            all_labels += labels.tolist()
            all_preds += preds.tolist()

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        results[group] = compute_results(all_labels, all_preds)

        if verbose:
            print('{} results'.format(group))
            pprint(results[group])

    return results
