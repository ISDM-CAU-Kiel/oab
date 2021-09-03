import sys

sys.path.append('./')
sys.path.append('../../')

import pytest
import numpy as np
from oab.data.classification_dataset import ClassificationDataset

# input labels

input_labels = np.array([4, 3, 0, 2, 4, 4, 3, 0, 2, 4, 1])
# np.nan is missing value
input_nan = np.array([
    [     4,    0.4,      4, np.nan, np.nan,      4],
    [   0.1,    0.4,      4,      1,    0.2,      4],
    [   0.1,    0.2,      3,      2, np.nan,      4],
    [   0.1,    0.2,      0,      1,    0.2,      3],
    [   0.1,    0.2,      0,      1,    0.2,      3],
    [   0.1, np.nan,      4,      1,    0.2,      4],
    [   0.1, np.nan,      4,      1,    0.2,      4],
    [   0.1,    0.2,      0,      1,    0.2,      2],
    [   0.1,    0.2, np.nan,      1,    0.2,      2],
    [np.nan,    0.2,      4,      1,    0.2,      4],
    [   0.1,    0.2,      0,      1,    0.2,      2],
])
result_nan = np.array([
    [   0.1,      4,      1,      4],
    [   0.1,      3,      2,      4],
    [   0.1,      0,      1,      3],
    [   0.1,      0,      1,      3],
    [   0.1,      4,      1,      4],
    [   0.1,      4,      1,      4],
    [   0.1,      0,      1,      2],
    [   0.1,      0,      1,      2],
])
result_labels_nan = np.array([3, 0, 2, 4, 4, 3, 0, 1])

# 4 is missing value
input_4 = np.array([
    [     4,    0.4,      2,      9,      9,      4],
    [   0.1,    0.4,      2,      1,    0.2,      4],
    [   0.1,    0.2,      2,      2,      9,      4],
    [   0.1,    0.2,      0,      1,    0.2,      3],
    [   0.1,    0.2,      0,      1,    0.2,      3],
    [   0.1,      9,      1,      1,    0.2,      4],
    [   0.1,      9,      1,      1,    0.2,      4],
    [   0.1,    0.2,      0,      1,    0.2,      2],
    [   0.1,    0.2,      9,      1,    0.2,      2],
    [     9,    0.2,      4,      1,    0.2,      4],
    [   0.1,    0.2,      0,      1,    0.2,      2],
])
result_4 = np.array([
    [   0.1,    0.4,      2,      1,    0.2],
    [   0.1,    0.2,      2,      2,      9],
    [   0.1,    0.2,      0,      1,    0.2],
    [   0.1,    0.2,      0,      1,    0.2],
    [   0.1,      9,      1,      1,    0.2],
    [   0.1,      9,      1,      1,    0.2],
    [   0.1,    0.2,      0,      1,    0.2],
    [   0.1,    0.2,      9,      1,    0.2],
    [   0.1,    0.2,      0,      1,    0.2],
])
result_labels_4 = np.array([3, 0, 2, 4, 4, 3, 0, 2, 1])

# some labels
labels = np.array([4, 3, 0, 2, 4, 4, 3, 0, 2, 4, 1])

@pytest.mark.parametrize(
    'input_values, expected_values, input_labels, expected_labels, missing_value', [
        (input_nan, result_nan, input_labels, result_labels_nan, np.nan),
        (  input_4,   result_4, input_labels,   result_labels_4,    4)
    ]
)
def test_treat_missing_values(input_values, expected_values,
        input_labels, expected_labels, missing_value):
    cd = ClassificationDataset(input_values, input_labels)
    cd.treat_missing_values(missing_value=missing_value)
    assert (np.all(np.isclose(cd.values, expected_values)) and np.all(np.isclose(cd.labels, expected_labels)))
