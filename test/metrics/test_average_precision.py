import sys

sys.path.append('./')
sys.path.append('../../')

import pytest
import numpy as np
from oab.metrics import average_precision_score, adjusted_average_precision_score

# contamination=.5, n=10, all correct
labels_5_10_all, preds_5_10_all = ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0.8, 0.2, 0.95, 0.1, 0.92, 0.13, 0.67, 0.44, 0.83, 0.04])
# contamination=.5, n=10, first 3 correct, then 3 wrong, then 2 correct
labels_5_10_1, preds_5_10_1, ap_5_10_1 = ([1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0.9, 0.85, 0.8, 0.4, 0.42, 0.65, 0.64, 0.6, 0.2, 0.1],
    (0.2 * (3 + 4/7 + 5/8)))
# contamination=.3, n=10, first 1 correct, then 1 wrong, then 2 correct
labels_3_10_1, preds_3_10_1, ap_3_10_1 = ([1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0.9, 0.645, 0.642, 0.32, 0.3, 0.65, 0.64, 0.6, 0.2, 0.1],
    (1/3 * (1 + 2/3 + 3/4)))


@pytest.mark.parametrize(
    'labels, predictions, expected', [
        (labels_3_10_1, preds_3_10_1, ap_3_10_1),
        (labels_5_10_1, preds_5_10_1, ap_5_10_1),
        (labels_5_10_all, preds_5_10_all, 1)
    ]
)
def test_average_precision_score(labels, predictions, expected):
    assert np.isclose(
        average_precision_score(y=labels, y_pred_scores=predictions),
        expected)



@pytest.mark.parametrize(
    'labels, predictions, expected', [
        (labels_3_10_1, preds_3_10_1, (ap_3_10_1 - .3)/.7),
        (labels_5_10_1, preds_5_10_1, (ap_5_10_1 - .5)/.5),
        (labels_5_10_all, preds_5_10_all, 1)
    ]
)
def test_adjusted_average_precision_score(labels, predictions, expected):
    assert np.isclose(
        adjusted_average_precision_score(y=labels, y_pred_scores=predictions),
        expected)
