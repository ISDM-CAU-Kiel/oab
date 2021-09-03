import pytest
import sys
import numpy as np

sys.path.append('./')
sys.path.append('../../')

from collections import Counter

from oab.data.classification_dataset import ClassificationDataset
from oab.data.semisupervised import SemisupervisedAnomalyDataset

values = np.array([
       [ 0.        ,  0.        , -1.23814117],
       [ 0.        ,  0.        , -0.34782977],
       [ 0.        ,  0.        , -0.58019045],
       [ 0.        ,  0.        , -0.30498998],
       [ 0.        ,  0.        ,  0.05997026],
       [ 0.        ,  0.        , -0.41473282],
       [ 0.        ,  0.        , -0.73909723],
       [ 0.        ,  0.        ,  0.53017559],
       [ 0.        ,  0.        , -0.45911861],
       [ 0.        ,  0.        , -1.70658038],
       [ 0.        ,  0.        ,  0.31575729],
       [ 0.        ,  0.        , -3.34318919],
       [ 0.        ,  0.        , -0.20704654],
       [ 0.        ,  0.        ,  0.66102655],
       [ 0.        ,  0.        , -0.66845186],
       [ 0.        ,  0.        , -0.04942478],
       [ 0.        ,  0.        , -0.58603792],
       [ 0.        ,  0.        ,  0.60308299],
       [ 0.        ,  0.        , -1.44704763],
       [ 0.        ,  0.        ,  0.25312138],
       [ 0.        ,  0.        , -0.32973997],
       [ 0.        ,  0.        ,  0.99283364],
       [ 0.        ,  0.        ,  0.03872232],
       [ 0.        ,  0.        ,  0.79573068],
       [ 0.        ,  0.        ,  0.87046742],
       [ 0.        ,  0.        ,  2.37208603],
       [ 0.        ,  0.        , -1.42377547],
       [ 0.        ,  0.        , -0.27484465],
       [ 0.        ,  0.        ,  0.68903344],
       [ 0.        ,  0.        ,  1.35536234],
       [ 1.        ,  5.        ,  1.91554064],
       [ 1.        ,  5.        , -1.34726297],
       [ 1.        ,  5.        , -0.186107  ],
       [ 1.        ,  5.        , -1.56729699],
       [ 1.        ,  5.        ,  1.42086084],
       [ 1.        ,  5.        ,  0.89225212],
       [ 1.        ,  5.        ,  0.31673233],
       [ 1.        ,  5.        , -0.47586681],
       [ 1.        ,  5.        ,  1.21577765],
       [ 1.        ,  5.        ,  0.61464497],
       [ 1.        ,  1.        , -0.15599474],
       [ 1.        ,  1.        , -0.4966991 ],
       [ 1.        ,  1.        , -0.16080949],
       [ 1.        ,  1.        ,  2.58853986],
       [ 1.        ,  1.        , -0.25578615],
       [ 1.        ,  1.        , -0.24439699],
       [ 1.        ,  1.        ,  1.21813455],
       [ 1.        ,  1.        ,  0.62547415],
       [ 1.        ,  1.        ,  0.44969056],
       [ 1.        ,  1.        ,  0.80994805],
       [ 1.        ,  1.        ,  1.43614542],
       [ 1.        ,  1.        ,  1.24629719],
       [ 1.        ,  1.        , -0.45903133],
       [ 1.        ,  1.        , -2.04471917],
       [ 1.        ,  1.        , -0.82605262],
       [ 1.        ,  1.        , -1.11926308],
       [ 1.        ,  1.        , -0.36926083],
       [ 1.        ,  1.        , -0.28121064],
       [ 1.        ,  1.        , -1.87104811],
       [ 1.        ,  1.        ,  2.15133942]])
labels = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2.])


### (1) TEST sample: correct number of training data, test data, contamination_rate
@pytest.mark.parametrize(
    'n_training, n_test, training_contamination_rate, n_training_normal,'\
    ' n_training_anomaly, test_contamination_rate, n_test_normals, n_test_anomalies', [
        (20, 20, 0.05, 19, 1, 0.1, 18, 2),
        (30, 10,    0, 30, 0, 0.1, 9,  1),
        (30, 20,  0.1, 27, 3, 0.2, 16, 4),
    ]
)
def test_contamination_rate(n_training, n_test,
    training_contamination_rate, n_training_normal, n_training_anomaly,
    test_contamination_rate, n_test_normals, n_test_anomalies):
    cd = ClassificationDataset(values=values, labels=labels, name="Test")
    ad = SemisupervisedAnomalyDataset(cd, normal_labels=[0, 1])

    (training_set, test_values, test_labels), _ = ad.sample(n_training=n_training,
        n_test=n_test,
        training_contamination_rate=training_contamination_rate,
        test_contamination_rate=test_contamination_rate)

    # check training set
    assert len(training_set) == n_training
    assert np.sum(np.isclose(training_set[:, 1], 1)) == n_training_anomaly
    assert np.sum(~ np.isclose(training_set[:, 1], 1)) == n_training_normal

    # check test set
    assert len(test_values) == n_test
    assert Counter(test_labels)[0] == n_test_normals
    assert Counter(test_labels)[1] == n_test_anomalies
    assert np.isclose((Counter(test_labels)[1]/len(test_values)), test_contamination_rate)



### (2) TEST sample: the values that are returned actually exist
@pytest.mark.parametrize(
    'n_training, n_test, training_contamination_rate, test_contamination_rate, random_seed', [
        (20, 20, 0.0, 0.2, 42),
        (30, 10, 0.0, 0.1, 20),
        (30, 20, 0.1, 0.5, 10),
    ]
)
def test_samples_exist(n_training, n_test, training_contamination_rate,
    test_contamination_rate, random_seed):
    cd = ClassificationDataset(values=values, labels=labels, name="Test")
    ad = SemisupervisedAnomalyDataset(cd, normal_labels=[0, 1])

    (training_set, test_values, test_labels), _ = ad.sample(n_training=n_training,
        n_test=n_test,
        training_contamination_rate=training_contamination_rate,
        test_contamination_rate=test_contamination_rate, random_seed=random_seed)

    # check that training data exists
    for value in training_set:
        arr_bools = np.all(values == value, axis=1)
        assert np.any(arr_bools)

    for value, label in zip(test_values, test_labels):
        arr_bools = np.all(values == value, axis=1)
        # check if the value acutually exists
        assert np.any(arr_bools)
        # check that it is indeed of the correct class
        idx = np.arange(len(labels))[arr_bools]
        # case 1: normal
        if np.isclose(label, 0):
            assert (np.isclose(labels[idx], label) or np.isclose(labels[idx], 1.0))
        # case 2: anomlay. Class either 1 or 2
        else:
            assert np.isclose(labels[idx], 2.0)



### (3) TEST sample: same seed leads to same results
@pytest.mark.parametrize(
    'n_training, n_test, training_contamination_rate, test_contamination_rate, random_seed', [
        (20, 20, 0.0, 0.2, 42),
        (30, 10, 0.0, 0.1, 20),
        (30, 20, 0.1, 0.5, 10),
    ]
)
def test_same_seed(n_training, n_test, training_contamination_rate,
    test_contamination_rate, random_seed):
    cd = ClassificationDataset(values=values, labels=labels, name="Test")
    ad = SemisupervisedAnomalyDataset(cd, normal_labels=[0, 1])

    (ts1, tv1, tl1), _ = ad.sample(n_training=n_training, n_test=n_test,
        training_contamination_rate=training_contamination_rate,
        test_contamination_rate=test_contamination_rate, random_seed=random_seed)

    (ts2, tv2, tl2), _ = ad.sample(n_training=n_training, n_test=n_test, 
        training_contamination_rate=training_contamination_rate,
        test_contamination_rate=test_contamination_rate, random_seed=random_seed)

    assert np.all(ts1 == ts2)
    assert np.all(tv1 == tv2)
    assert np.all(tl1 == tl2)
