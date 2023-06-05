import bisect
import pickle
from typing import List

import numpy as np


def type_acc_np(preds, labels, **kwargs):
    """ Type accuracy ratio  """
    type_pred = np.array(preds) if isinstance(preds, list) else preds
    type_label = np.array(labels) if isinstance(labels, list) else labels

    return np.mean(type_pred == type_label)


def time_rmse_np(preds, labels, **kwargs):
    """ RMSE for time predictions """
    preds = np.array(preds) if isinstance(preds, list) else preds
    labels = np.array(labels) if isinstance(labels, list) else labels
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    return rmse


def is_hit(label, pred, top_n=5):
    """

    Args:
        label: [batch_size]
        pred: [batch_size, num_candidate]
        top_n: int

    Returns:

    """
    sort_index = np.argsort(pred)[:, -top_n:]
    batch_size = len(label)
    res = []
    for i in range(batch_size):
        label_i = label[i]
        res.append(label_i in sort_index[i, :])

    return np.array(res)


def rank(label, pred):
    """

    Args:
        label: [N]
            Label index.
        pred: [N, n_cat]
            Predicted values.

    Returns:

    """
    label = np.array(label)
    pred = np.array(pred)
    if len(label.shape) < 2:
        label = label[:, None]
    if len(pred.shape) > 2:
        pred = pred.reshape(pred.shape[0], pred.shape[-1])

    label_val = np.take_along_axis(pred, label, axis=1)
    big_num = (pred > label_val).sum(axis=-1)
    equal_num = np.maximum((pred == label_val).sum(axis=-1), 1)

    rank = big_num + equal_num
    return rank


def get_obj_rel_list_at_given_sub_time(source_data, sub, time):
    obj_list = [x[1] for x in source_data if x[0] == sub and x[3] == time]
    rel_list = [x[2] for x in source_data if x[0] == sub and x[3] == time]
    return obj_list, rel_list


def group_pred_data_in_time(pred_data, source_data):
    res = []
    for pred in pred_data:
        idx = pred['original_idx']
        sub, obj, rel, time, _ = source_data[int(idx)]
        obj_list, rel_list = get_obj_rel_list_at_given_sub_time(source_data, sub, time)
        res_i = dict({'original_idx': idx,
                      'subject': sub,
                      'object_list': obj_list,
                      'relation_list': rel_list,
                      'sorted_hl_object': np.argsort(pred['pred_object'].flatten())[::-1].tolist(),  # from highest to lowest
                      'sorted_hl_relation': np.argsort(pred['pred_relation'].flatten())[::-1].tolist()})

        res.append(res_i)

    return res


#######################  Mean Average Recall and Precision ################################
#############  https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py  #######


def _precision(predicted, actual):
    prec = [value for value in predicted if value in actual]
    prec = float(len(prec)) / float(len(predicted))
    return prec


def _ark(actual: list, predicted: list, k=10) -> float:
    """
    Computes the average recall at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : float
        The average recall at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / len(actual)


def _apk(actual: list, predicted: list, k=10) -> float:
    """
    Computes the average precision at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : float
        The average precision at k.
    """
    if not predicted or not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    true_positives = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            max_ix = min(i + 1, len(predicted))
            score += _precision(predicted[:max_ix], actual)
            true_positives += 1

    if score == 0.0:
        return 0.0

    return score / true_positives


def mark(actual: List[list], predicted: List[list], k=10) -> float:
    """
    Computes the mean average recall at k.
    Parameters
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        mark: float
            The mean average recall at k (mar@k)
    """
    if len(actual) != len(predicted):
        raise AssertionError("Length mismatched")

    return np.mean([_ark(a, p, k) for a, p in zip(actual, predicted)])


def mapk(actual: List[list], predicted: List[list], k: int = 10) -> float:
    """
    Computes the mean average precision at k.
    Parameters
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        mark: float
            The mean average precision at k (map@k)
    """
    if len(actual) != len(predicted):
        raise AssertionError("Length mismatched")

    return np.mean([_apk(a, p, k) for a, p in zip(actual, predicted)])


def get_precision_recall(pred_dict, rel_topk, obj_topk):
    res = dict()

    all_object_pred = [x['sorted_hl_object'] for x in pred_dict]
    all_rel_pred = [x['sorted_hl_relation'] for x in pred_dict]

    all_object_label = [set(x['object_list']) for x in pred_dict]
    all_rel_label = [set(x['relation_list']) for x in pred_dict]

    res[f'mar_object_{obj_topk}'] = mark(actual=all_object_label, predicted=all_object_pred, k=obj_topk)
    res[f'mar_rel_{rel_topk}'] = mark(actual=all_rel_label, predicted=all_rel_pred, k=rel_topk)

    res[f'map_object_{obj_topk}'] = mapk(actual=all_object_label, predicted=all_object_pred, k=obj_topk)
    res[f'map_rel_{rel_topk}'] = mapk(actual=all_rel_label, predicted=all_rel_pred, k=rel_topk)

    return res


def get_target_pos_index(pred_target='relation'):
    if pred_target.lower() in ['relation', 'rel', 'pred_relation']:
        return 2
    elif pred_target.lower() in ['object', 'obj', 'pred_object']:
        return 1
    elif pred_target.lower() in ['time', 't', 'pred_time']:
        return 3
    else:
        raise NotImplementedError


def get_pct_ratio(input_list):
    res = {}
    unique_value = set(input_list)
    for i in unique_value:
        res[i] = input_list.count(i) / len(input_list)
    return res
