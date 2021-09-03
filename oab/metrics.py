import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
from sklearn.metrics import average_precision_score as sklearn_average_precision_score
from pyod.utils.utility import precision_n_scores as pyod_precision_n_scores

from typing import Dict

def roc_auc_score(y: np.ndarray, y_pred_scores: np.ndarray, **kwargs: Dict) -> float:
    """Calculates ROC AUC score for an anomaly prediction. Wrapper for sklearn's
    ROC AUC score.

    :param y: Binary ground truth values (0: normals, 1: anomalies)
    :param y_pred_scores: The raw anomaly scores as returned by a fitted model
    :param kwargs: Dictionary of keyword arguments passed to sklearn's
        implementation

    :return: ROC AUC score
    """
    return sklearn_roc_auc_score(y, y_pred_scores, **kwargs)


def precision_n_score(y: np.ndarray, y_pred_scores: np.ndarray,
        n:int = None) -> float:
    """Calculate precision @ n for an anomaly perediction.

    :param y: Binary ground truth values (0: normals, 1: anomalies)
    :param y_pred_scores: The raw anomaly scores as returned by a fitted model
    :param n: Top n scores are considered when calculating the precision, and this
        specifies n. If not set, defaults to the number of outliers in the
        ground truth, defaults to None

    :return: Precision @ n score
    """
    # TODO: does it have to be numpy array? Can we also do tf. and torch?
    # TODO write tests
    # (TODO) assert that arguments are valid to raise helpful errors
    return pyod_precision_n_scores(y, y_pred_scores, n=n)



def adjusted_precision_n_score(y: np.ndarray, y_pred_scores: np.ndarray,
        n: int = None) -> float:
    """
    Calculate precision @ n adjusted for chance.

    :param y: Binary ground truth values (0: normals, 1: anomalies)
    :param y_pred_scores: The raw anomaly scores as returned by a fitted model
    :param n: Top n scores are considered when calculating adjusted precision,
        and this specifies n. If not set, defaults to the number of outliers in the
        ground truth, defaults to None

    :return: Adjusted precision @ n score
    """

    # TODO: check for matching sizes. Might actually do that in a util function
    if type(y_pred_scores) == list:
        number_instances = len(y_pred_scores)
    else:
        number_instances = y_pred_scores.shape[0]
    number_outliers = np.count_nonzero(y)
    contamination_rate = number_outliers / number_instances

    # if n is not specified, set n to the number of outliers in ground truth
    if n == None:
        n = number_outliers

    prec_n = precision_n_score(y, y_pred_scores, n=n)

    # n <= |O|: Here, 1 is the maximum precision @ n score
    if n <= number_outliers:
        return (prec_n - contamination_rate) / (1 - contamination_rate)

    # n > |O|: Here, |O|/n is the maximum precision @ n score
    else:
        return (prec_n - contamination_rate) / ((number_outliers / n) - contamination_rate)



def average_precision_score(y: np.ndarray, y_pred_scores: np.ndarray) -> float:
    """
    Calculates the average precision, i.e., 1/|O| * \sum_{o \in O}{Precision @ rank(o)}
    where O is the set of outliers, and rank(o) (o \in O) is the outlier-rank
    assigned to o, where a higher rank means that it is more likely an outlier.
    Wrapper for sklearn's average precision score.

    :param y: Binary ground truth values (0: normals, 1: anomalies)
    :param y_pred_scores: The raw anomaly scores as returned by a fitted model

    :return: Average precision score
    """
    return sklearn_average_precision_score(y, y_pred_scores)



def adjusted_average_precision_score(y: np.ndarray,
        y_pred_scores: np.ndarray) -> float:
    """
    Calculates the adjusted average precision, where adjustments for chance are
    made, i.e., (AP - |O|/N) / (1 - |O|/N),
    where AP is the average precision, O is the set of outliers and N is the
    number of samples.

    :param y: Binary ground truth values (0: normals, 1: anomalies)
    :param y_pred_scores: The raw anomaly scores as returned by a fitted model

    :return: Average precision score
    """
    ap = average_precision_score(y, y_pred_scores)
    # TODO: check for matching sizes. Might actually do that in a util function
    if type(y_pred_scores) == list:
        number_instances = len(y_pred_scores)
    else:
        number_instances = y_pred_scores.shape[0]
    number_outliers = np.count_nonzero(y)
    contamination_rate = number_outliers / number_instances

    return (ap - contamination_rate) / (1 - contamination_rate)


def precision_recall_auc_score(y: np.ndarray, y_pred_scores: np.ndarray) -> float:
    """
    Calculates the area under precision recall curve.

    :param y: Binary ground truth values (0: normals, 1: anomalies)
    :param y_pred_scores: The raw anomaly scores as returned by a fitted model

    :return: Area under precision recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true=y,
        probas_pred=y_pred_scores)
    return auc(recall, precision)
