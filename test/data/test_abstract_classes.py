# IDEA: Use Unsupervised (and Semisupervised) AnomalyDataset to test functionality provided by the abstract class AnomalyDataset

import pytest
import sys
import numpy as np

sys.path.append('./')
sys.path.append('../../')

from oab.data.classification_dataset import ClassificationDataset
from oab.data.unsupervised import UnsupervisedAnomalyDataset


### (1) TEST _get_all_other_labels
values = np.array([[1.0, 2.0, 3.0]] * 10)
labels = np.array([1, 1, 1, 2, 2, 2, 3, 4, 5, 6])

@pytest.mark.parametrize(
    'normal_labels, anomaly_labels', [
        ([1], [2, 3, 4, 5, 6]),
        ([1, 2], [3, 4, 5, 6]),
        ([3, 4, 5], [1, 2, 6])
    ]
)
def test_get_all_other_labels(normal_labels, anomaly_labels):
    cd = ClassificationDataset(values=values, labels=labels, name="Test")
    ad = UnsupervisedAnomalyDataset(cd, normal_labels=normal_labels)
    assert set(ad.anomaly_labels) == set(anomaly_labels)



### (2) TEST _get_idxs
@pytest.mark.parametrize(
    'normal_labels, all_idxs_normal, all_idxs_anomaly', [
        ([1], [0, 1, 2], [3, 4, 5, 6, 7, 8, 9]),
        ([2], [3, 4, 5], [0, 1, 2, 6, 7, 8, 9]),
        ([3, 5], [6, 8], [0, 1, 2, 3, 4, 5, 7, 9])
    ]
)
def test_get_idxs_all(normal_labels, all_idxs_normal, all_idxs_anomaly):
    cd = ClassificationDataset(values=values, labels=labels, name="Test")
    ad = UnsupervisedAnomalyDataset(cd, normal_labels=normal_labels)
    assert set(ad.normal_idxs) == set(all_idxs_normal)
    assert set(ad.anomaly_idxs) == set(all_idxs_anomaly)


@pytest.mark.parametrize(
    'normal_labels, idxs_0, idxs_1', [
        ([1, 2], [0, 1, 2], [3, 4, 5]),
        ([2, 5], [3, 4, 5], [8]),
        ([3, 5], [6], [8])
    ]
)
def test_get_idxs_individual(normal_labels, idxs_0, idxs_1):
    cd = ClassificationDataset(values=values, labels=labels, name="Test")
    ad = UnsupervisedAnomalyDataset(cd, normal_labels=normal_labels)
    assert set(ad.normal_idxs_individual[0]) == set(idxs_0)
    assert set(ad.normal_idxs_individual[1]) == set(idxs_1)
