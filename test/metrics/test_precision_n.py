import sys

sys.path.append('./')
sys.path.append('../../')

import pytest
import numpy as np
from oab.metrics import precision_n_score, adjusted_precision_n_score

# contamination=.5, n=10, all correct
labels_5_10_all, preds_5_10_all = ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0.8, 0.2, 0.95, 0.1, 0.92, 0.13, 0.67, 0.44, 0.83, 0.04])
# contamination=.5, n=10, first 3 correct, then 3 wrong, then 2 correct
labels_5_10_1, preds_5_10_1 = ([1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0.9, 0.85, 0.8, 0.4, 0.42, 0.65, 0.64, 0.6, 0.2, 0.1])
# contamination=.3, n=10, first 1 correct, then 1 wrong, then 2 correct
labels_3_10_1, preds_3_10_1 = ([1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0.9, 0.645, 0.642, 0.32, 0.3, 0.65, 0.64, 0.6, 0.2, 0.1])

@pytest.mark.parametrize(
    'labels, predictions, n, expected', [
        ([1, 1, 0], [0.5, 0.4, 0.1], None, 1), # simplest case
        ([1, 1, 0], [0.5, 0.4, 0.1], 2, 1),
        ([1, 1, 0], [0.5, 0.4, 0.1], 1, 1),
        ([1, 1, 0, 0], [0.5, 0.4, 0.1, 0.2], 3, 2/3),

        (labels_5_10_all, preds_5_10_all, 5, 1),
        (labels_5_10_all, preds_5_10_all, None, 1),
        (labels_5_10_all, preds_5_10_all, 3, 1),
        (labels_5_10_all, preds_5_10_all, 8, 5/8),

        (labels_5_10_1, preds_5_10_1, None, 3/5),
        (labels_5_10_1, preds_5_10_1, 3, 1),
        (labels_5_10_1, preds_5_10_1, 4, 3/4),

        (labels_3_10_1, preds_3_10_1, 1, 1),
        (labels_3_10_1, preds_3_10_1, None, 2/3),
        (labels_3_10_1, preds_3_10_1, 5, 3/5)
    ]
)
def test_precision_n_score(labels, predictions, n, expected):
    assert np.isclose(precision_n_score(y=labels, y_pred_scores=predictions, n=n), expected)
    # assert precision_n_score(y=labels, y_pred_scores=predictions, n=n) == expected


@pytest.mark.parametrize(
    'labels, predictions, n, expected', [
        ([1, 1, 0, 0], [0.5, 0.4, 0.1, 0.2], 2, 1),

        (labels_5_10_all, preds_5_10_all, 5, 1),
        (labels_5_10_all, preds_5_10_all, 3, 1),
        (labels_5_10_all, preds_5_10_all, 8, 1),

        (labels_5_10_1, preds_5_10_1, None, 0.2),
        (labels_5_10_1, preds_5_10_1, 3, 1),
        (labels_5_10_1, preds_5_10_1, 4, 1/2),

        (labels_3_10_1, preds_3_10_1, None, (2/3 - 0.3) / 0.7),
        (labels_3_10_1, preds_3_10_1, 5, 1)
    ]
)
def test_adjusted_precision_n_score(labels, predictions, n, expected):
    assert np.isclose(adjusted_precision_n_score(y=labels, y_pred_scores=predictions, n=n), expected)
    # assert precision_n_score(y=labels, y_pred_scores=predictions, n=n) == expected
    # assert precision_n_score([1, 1, 0], [0.5,0.4, 0.1]) == 1
