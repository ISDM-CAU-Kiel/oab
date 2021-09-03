import pytest
import sys
import numpy as np

sys.path.append('./')
sys.path.append('../../')

from collections import Counter

from oab.data.classification_dataset import ClassificationDataset
from oab.data.unsupervised import UnsupervisedAnomalyDataset

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

### (1) TEST sample: correct contamination rate (in random sampling) and number of instances
@pytest.mark.parametrize(
    'n, contamination_rate, expected_n_normals, expected_n_anomalies', [
        (50, 0.4, 30, 20),
        (10, 0.1,  9,  1),
        (20, 0.1, 18,  2)
    ]
)
def test_contamination_rate(n, contamination_rate, expected_n_normals, expected_n_anomalies):
    cd = ClassificationDataset(values=values, labels=labels, name="Test")
    ad = UnsupervisedAnomalyDataset(cd, normal_labels=[0])
    (_, test_labels), _ = ad.sample(n, contamination_rate=contamination_rate)
    assert len(test_labels) == n
    assert Counter(test_labels)[0] == expected_n_normals
    assert Counter(test_labels)[1] == expected_n_anomalies



### (2) TEST sample: all instances that are output were actually in the input
@pytest.mark.parametrize(
    'random_seed', [
        (42),
        (18),
        (10005)
    ]
)
def test_samples_exist(random_seed):
    n = 30
    contamination_rate = 0.2

    cd = ClassificationDataset(values=values, labels=labels, name="Test")
    ad = UnsupervisedAnomalyDataset(cd, normal_labels=[0])

    (sampled_values, sampled_labels), _ = ad.sample(n,
        contamination_rate=contamination_rate, random_seed=random_seed)

    for value, label in zip(sampled_values, sampled_labels):
        arr_bools = np.all(values == value, axis=1)
        # check if the value acutually exists
        assert np.any(arr_bools)
        # check that it is indeed of the correct class
        idx = np.arange(len(labels))[arr_bools]
        # case 1: normal
        if np.isclose(label, 0):
            assert np.isclose(labels[idx], label)
        # case 2: anomlay. Class either 1 or 2
        else:
            assert (np.isclose(labels[idx], 1.0) or np.isclose(labels[idx], 2.0))



### (3) TEST sample: sampling with the same random seed gives the same result
@pytest.mark.parametrize(
    'random_seed', [
        (42),
        (18),
        (10005)
    ]
)
def test_same_seed(random_seed):
    n = 30
    contamination_rate = 0.2

    cd = ClassificationDataset(values=values, labels=labels, name="Test")
    ad = UnsupervisedAnomalyDataset(cd, normal_labels=[0])

    (sv1, sl1), _ = ad.sample(n, contamination_rate=contamination_rate,
        random_seed=random_seed)
    (sv2, sl2), _ = ad.sample(n, contamination_rate=contamination_rate,
        random_seed=random_seed)

    assert np.all(sv1 == sv2)
    assert np.all(sl1 == sl2)



# TODO Test: corect contamination rate (when keeping ratios/equal frequency)
