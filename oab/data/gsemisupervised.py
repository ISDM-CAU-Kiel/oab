import numpy as np
import pandas as pd

import torch
from typing import Union, Tuple, Optional
from dataclasses import dataclass
from oab.data.abstract_classes import AnomalyDataset, AnomalyDatasetDescription, AbstractClassificationDataset
from oab.data.utils import _append_to_yaml, _make_yaml, _append_sampling_to_yaml
from oab.data.utils_image import image_datasets, reshape_images

import numpy as np
from sklearn.model_selection import train_test_split
import random

def train_test_split_(ratio, indexes, lst=False):
    import math
    n_trn = math.ceil(ratio*len(indexes))
    if lst:
        idx_trn, idx_tst = list(indexes[:n_trn]), list(indexes[n_trn:])
    else:
        idx_trn, idx_tst = np.array(indexes[:n_trn]), np.array(indexes[n_trn:])
    return idx_trn, idx_tst


@dataclass
class GSemisupervisedAnomalyDatasetDescription(AnomalyDatasetDescription):
    """Description object for a sample from a general semisupervised anomaly dataset.

    :param name: Name of the dataset
    :param normal_labels: List of normal labels
    :param anomaly_labels: List of anomalous labels
    :param number_instances_training: Number of instances in the training set
    :param number_instances_test: Number of instances in the test set
    :param training_number_normals: Number of normal data points in the training set
    :param training_number_anomalies: Number of anomalous data points in the training set
    :param training_contamination_rate: Contamination rate in the training set
    :param test_number_normals: Number of normal data points in the test set
    :param test_number_anomalies: Number of anomalous data points in the test set
    :param test_contamination_rate: Contamination rate in the test set
    """
    name: str
        
    normal_labels: list()   
    normal_classes: list() # should contain same as normal_labels 
        
    anomaly_labels: list()
    outlier_classes: list() # should contain same as anomaly_labels
        
    known_outlier_class: int
    n_known_outlier_classes: int
        
    ratio_known_normal: float
    ratio_known_outlier: float
    ratio_pollution: float
    
    number_instances_training: int
    number_instances_test: int

    training_number_normals: int
    training_number_anomalies: int
    training_contamination_rate: float

    test_number_normals: int
    test_number_anomalies: int
    test_contamination_rate: float
        
    #################
    
    # https://github.com/lukasruff/Deep-SAD-PyTorch/blob/2e7aca37412e7f09d42d48d9e722ddfb422c814a/src/base/base_dataset.py
        
    #todo:
    #normal_classes = None  # tuple with original class labels that define the normal class
    #outlier_classes = None  # tuple with original class labels that define the outlier class
    #train_set = None  # must be of type torch.utils.data.Dataset
    #test_set = None  # must be of type torch.utils.data.Dataset
    
    #################



    def from_same_dataset(self, other: 'GSemisupervisedAnomalyDatasetDescription') -> bool:
        """
        Test if two dataset descriptions come from a similar sampling fo the same
        anomlay dataset.

        :param other: Other dataset description
        :return: True if the two datasets come from a similar sampling (in terms of contamination_rate and number_instances)
            from the same anomaly dataset (in terms of name, normal_labels, anomaly_labels).
            False otherwise.

        """
        # If we have image datasets, they don't need to have the same anomaly
        # labels (as all labels are anomalie once)
        if self.name in image_datasets:
            self_attributes = (self.name,
                               self.number_instances_training,
                               self.number_instances_test,
                               )
            other_attributes = (other.name,
                                other.number_instances_training,
                                other.number_instances_test,
                                )
        else:
            self_attributes = (self.name,
                               self.number_instances_training,
                               self.number_instances_test,
                               set(self.normal_labels),
                               set(self.anomaly_labels))
            other_attributes = (other.name,
                                other.number_instances_training,
                                other.number_instances_test,
                                set(other.normal_labels),
                                set(other.anomaly_labels))
        float_self_attributes = (self.training_contamination_rate,
                                 self.test_contamination_rate)
        float_other_attributes = (other.training_contamination_rate,
                                  other.test_contamination_rate)
        float_comparison = np.all(np.isclose(float_self_attributes, float_other_attributes))
        return (self_attributes == other_attributes) and float_comparison


    def print_for_eval_specifics(self) -> str:
        """Return pretty string representation of most important dataset characteristics.

        :return: String with the most important dataset characteristics
        """
        return (f"{self.number_instances_training} training instances, " \
                f"{self.number_instances_test} test instances, " \
                f"training contamination rate {self.training_contamination_rate}, " \
                f"test contamination rate {self.test_contamination_rate}.")
    

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, classification_dataset: AbstractClassificationDataset, transform=None, target_transform=None):
        
        self.data = torch.tensor(classification_dataset.values)
        self.targets = torch.tensor(classification_dataset.labels, dtype=torch.int64)
        self.semi_targets = torch.zeros_like(self.targets, dtype=torch.int64)
        
        self.transform = transform # check if it is not yet already done previously
        self.target_transform = target_transform # check if it is not yet already done previously
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        
        image, target, semi_target = self.data[idx], int(self.targets[idx]), int(self.semi_targets[idx])
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # image = Image.fromarray(image.numpy(), mode='L') #todo: verify if its really necessary or already done before
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target, semi_target, idx

    
class GSemisupervisedAnomalyDataset(AnomalyDataset):
    """This class represents a general semisupervised anomaly dataset, i.e., when
    sampling from the dataset, an array of (potentially polluted or just partially labeled) values is returned for
    training, and a tuple for testing with values and labels of the observation
    are returned.
    """
    def __init__(self, **kwargs):
        super(GSemisupervisedAnomalyDataset, self).__init__(**kwargs)
        
        self.normal_classes = self.normal_labels
        self.outlier_classes = self.anomaly_labels
            
    def _get_sample_pool_sizes(self, n_normal, ratio_known_normal, ratio_known_outlier, ratio_pollution, verbose=False):
        # Solve system of linear equations to obtain respective number of samples
        a = np.array([[1, 1, 0, 0],
                      [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                      [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                      [0, -ratio_pollution, (1-ratio_pollution), 0]])
        b = np.array([n_normal, 0, 0, 0])
        x = np.linalg.solve(a, b)

        # Get number of samples
        n_known_normal = int(x[0])
        n_unlabeled_normal = int(x[1])
        n_unlabeled_outlier = int(x[2])
        n_known_outlier = int(x[3])

        if verbose:
            print(f"n_total: {len(labels)}")
            print(f"n_known_normal: {n_known_normal}")
            print(f"n_unlabeled_normal: {n_unlabeled_normal}")
            print(f"n_unlabeled_outlier: {n_unlabeled_outlier}")
            print(f"n_known_outlier: {n_known_outlier}")

        return n_known_normal, n_unlabeled_normal, n_unlabeled_outlier, n_known_outlier
    
    def sample(self, n_training: int, n_test: int,
               training_contamination_rate: float = 0,
               test_contamination_rate: float = 0.2,
               shuffle: bool = True, random_seed: float = 42,
               apply_random_seed: bool =True,
               training_keep_frequency_ratio_normals: bool = False,
               training_equal_frequency_normals: bool = False,
               training_keep_frequency_ratio_anomalies: bool = False,
               training_equal_frequency_anomalies: bool = False,
               test_keep_frequency_ratio_normals: bool = False,
               test_equal_frequency_normals: bool = False,
               test_keep_frequency_ratio_anomalies: bool = False,
               test_equal_frequency_anomalies: bool = False,
               include_description: bool = True, flatten_images: bool = True,
               verbose: bool = False):
        """Sample from the anomaly dataset.
        Note that this method does not ensure that data points seen during
        training are resampled during testing. If possible, use other methods
        like sample_with_training_split or sample_with_explicit_numbers
        that ensure this.

        :param n_training: Number of training data points to sample
        :param n_test: Number of test data points to sample
        :param training_contamination_rate: Contamination rate when sampling
            training points, defaults to 0
        :param test_contamination_rate: Contamination rate when sampling
            test points, defaults to 0
        :param shuffle: Shuffle dataset, defaults to True
        :param random_seed: Random seed, defaults to 42
        :param apply_random_seed: Apply random seed to make sampling reproducible, defaults to True
        :param training_keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling for training. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points for training are sampled, 200 of them will be from
            label 0 and 100 from label 1. Defaults to False
        :param training_equal_frequency_normals: If there are multiple normal labels and
             this is set, training data will be sampled with an equal distribution
             among these normal labels, defaults to False
        :param training_keep_frequency_ratio_anomalies: Like parameter
            `training_keep_frequency_ratio_normals` for anomalies in training,
            defaults to False
        :param training_equal_frequency_anomalies: Like parameter
            `training_equal_frequency_normals` for anomalies in training,
            defaults to False
        :param test_keep_frequency_ratio_normals: Like parameter
            `training_keep_frequency_ratio_normals` for normals in test data,
            defaults to False
        :param test_equal_frequency_normals: Like parameter
            `training_equal_frequency_normals` for normals in test data,
            defaults to False
        :param test_keep_frequency_ratio_anomalies: Like parameter
            `training_keep_frequency_ratio_normals` for anomalies in test data,
            defaults to False
        :param test_equal_frequency_anomalies: Like parameter
            `training_equal_frequency_normals` for anomalies in test data,
            defaults to False
        :include_description: Includes sampling config file, defaults to True
        :param flatten_images: This flag has behavior only if an image dataset
            is loaded. If this is the case and this is set to True, the
            image will be flattened. If it is set to False, the original dimensionality
            of the image will be used, e.g., 32x32x3. Defaults to True

        :return: A tuple (x_train, x_test, y_test), sample_config. y_test and
            sample_config can be used to pass to the EvaluationObject.
        """
        self._test_contamination_rate(training_contamination_rate)
        self._test_contamination_rate(test_contamination_rate)
        if apply_random_seed:
            np.random.seed(random_seed)

        # # training set - clean
        # if np.isclose(0, training_contamination_rate):
        #     training_values, _, training_original_labels = \
        #         self._sample_data('normals', n_training, training_keep_frequency_ratio_normals,
        #                           training_equal_frequency_normals)
        #     n_training_normals = n_training
        #     n_training_anomalies = 0

        # training set
        # else:
        n_training_normals = int(n_training * (1 - training_contamination_rate))
        n_training_anomalies = n_training - n_training_normals

        training_values_normals, _, training_original_labels_normals = \
            self._sample_data('normals', n_training_normals, training_keep_frequency_ratio_normals,
                              training_equal_frequency_normals)
        training_values_anomalies, _, training_oringinal_labels_anomalies = \
            self._sample_data('anomalies', n_training_anomalies, training_keep_frequency_ratio_anomalies,
                              training_equal_frequency_anomalies)

        training_values = np.vstack((training_values_normals, training_values_anomalies))

        # test set
        n_test_normals = round(n_test * (1 - test_contamination_rate))
        n_test_anomalies = n_test - n_test_normals

        test_values_normals, test_labels_normals, test_original_labels_normals = \
            self._sample_data('normals', n_test_normals, test_keep_frequency_ratio_normals,
                              test_equal_frequency_normals)
        test_values_anomalies, test_labels_anomalies, test_original_labels_anomalies = \
            self._sample_data('anomalies', n_test_anomalies, test_keep_frequency_ratio_anomalies,
                              test_equal_frequency_anomalies)

        test_values = np.vstack((test_values_normals, test_values_anomalies))
        test_labels = np.hstack((test_labels_normals, test_labels_anomalies))

        # data description
        description = GSemisupervisedAnomalyDatasetDescription(name=self.classification_dataset.name,
            normal_labels=self.normal_labels, anomaly_labels=self.anomaly_labels,
            number_instances_training=n_training,
            number_instances_test=n_test,
            training_number_normals=n_training_normals,
            training_number_anomalies=n_training_anomalies,
            training_contamination_rate=training_contamination_rate,
            test_number_normals=n_test_normals, test_number_anomalies=n_test_anomalies,
            test_contamination_rate=test_contamination_rate)

        # shuffle
        if shuffle:
            # training set
            training_idxs = np.arange(n_training)
            np.random.shuffle(training_idxs)
            training_values = training_values[training_idxs]
            # test set
            test_idxs = np.arange(n_test)
            np.random.shuffle(test_idxs)
            test_values = test_values[test_idxs]
            test_labels = test_labels[test_idxs]

        # potentially reshape values if dealing with image data
        if self.classification_dataset.name in image_datasets:
            training_values = reshape_images(training_values,
                self.classification_dataset.name, flatten_images)
            test_values = reshape_images(test_values,
                self.classification_dataset.name, flatten_images)

        return (training_values, test_values, test_labels), description

    def sample_with_explicit_numbers(self, 
            training_normals_known: int, # n_known_normal
            training_anomalies_known: int, # n_known_outlier
            training_normals_unknown: int, # n_unlabeled_normal
            training_anomalies_unknown: int, # n_unlabeled_outlier
            test_normals: int, 
            test_anomalies: int,
            shuffle: bool = True, random_seed : float = 42,
            apply_random_seed: bool = True,
            keep_frequency_ratio_normals: bool = False,
            equal_frequency_normals: bool = False,
            keep_frequency_ratio_anomalies: bool = False,
            equal_frequency_anomalies: bool = False,
            include_description: bool = True,
            yamlpath_append: Optional = None, yamlpath_new: Optional = None,
            flatten_images: bool = True, 
            verbose: bool = False):
        """
        Sample specified number of points from the anomaly dataset.
        Note that both normal and anomaly points cannot occur in both
        training and test data.

        :param training_normals: Number of normal data points in the training set
        :param training_anomalies: Number of anomalies in the training set
        :param test_normals: Number of normal data points in the test set
        :param test_anomalies: Number of anomalies in the test set
        :param shuffle: Shuffle the training and test set, defaults to True
        :param random_seed: Seed for random number generator, defaults to 42
        :param apply_random_seed: Whether or not to apply random seed, defaults to True
        :param keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points are sampled, 200 of them will be from
            label 0 and 100 from label 1. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param equal_frequency_normals: If there are multiple normal labels and
             this is set, data will be sampled with an equal distribution
             among these normal labels. Note that the split of these values
             between train and test set is at random. Defaults to False
        :param keep_frequency_ratio_anomalies: If there are multiple anomaly labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if anomaly labels are [2, 3] and there
            are 2000 data points with label 2, 1000 data points with label 3 and
            300 anomalous data points are sampled, 200 of them will be from
            label 2 and 100 from label 3. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param equal_frequency_anomalies: If there are multiple anomaly labels and
             this is set, data will be sampled with an equal distribution
            among these anomaly labels. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param include_description: Whether or not to include a description of the
            sampled dataset, defaults to True
        :param yamlpath_append: Optionally append sampling arguments to a YAML
            if this is not None, the path of the YAML is specified in this
            argument, defaults to None
        :param yamlpath_new: Optionally create a new YAML with the sampling
            arguments if this is not None, the path of the YAML is specified in
            this argument, defaults to None
        :param flatten_images: This flag has behavior only if an image dataset
            is loaded. If this is the case and this is set to True, the
            image will be flattened. If it is set to False, the original dimensionality
            of the image will be used, e.g., 32x32x3. Defaults to True

        :return: A tuple of (training_values, test_values, test_labels), description
            where values and labels are a numpy.ndarray
        """
        kwargs = {'training_normals': training_normals,
            'training_anomalies': training_anomalies,
            'test_normals': test_normals, 'test_anomalies': test_anomalies,
            'shuffle': shuffle, 'random_seed': random_seed,
            'apply_random_seed': apply_random_seed,
            'include_description': include_description,
            'flatten_images': flatten_images
            }
        # store arguments in YAML if specified
        if yamlpath_append: # append arguments to YAML
            _append_sampling_to_yaml(yamlpath_append, "semisupervised_explicit_numbers_single", kwargs)
        if yamlpath_new: # create new YAML with arguments
            _make_yaml(yamlpath_new, "sampling", {'semisupervised_explicit_numbers_single': kwargs})

        if apply_random_seed:
            np.random.seed(random_seed)

        n_normals = training_normals + test_normals
        n_anomalies = training_anomalies + test_anomalies

        n_training = training_normals + training_anomalies
        n_test = test_normals + test_anomalies

        # sample data
        values_normals, labels_normals, original_labels_normals = \
            self._sample_data('normals', n_normals,
                keep_frequency_ratio=keep_frequency_ratio_normals,
                equal_frequency=equal_frequency_normals)
        values_anomalies, labels_anomalies, oringinal_labels_anomalies = \
            self._sample_data('anomalies', n_anomalies,
                keep_frequency_ratio=keep_frequency_ratio_anomalies,
                equal_frequency=equal_frequency_anomalies)

        # build training and test set
        training_values = np.vstack((values_normals[:training_normals],
                                     values_anomalies[:training_anomalies]))
        test_values = np.vstack((values_normals[training_normals:],
                                 values_anomalies[training_anomalies:]))
        test_labels = np.hstack((labels_normals[training_normals:],
                                 labels_anomalies[training_anomalies:]))


        # data description
        description = GSemisupervisedAnomalyDatasetDescription(name=self.classification_dataset.name,
            normal_labels=self.normal_labels, anomaly_labels=self.anomaly_labels,
            number_instances_training=training_normals + training_anomalies,
            number_instances_test=test_normals + test_anomalies,
            training_number_normals=training_normals,
            training_number_anomalies=training_anomalies,
            training_contamination_rate=training_anomalies/n_training,
            test_number_normals=test_normals,
            test_number_anomalies=test_anomalies,
            test_contamination_rate=test_anomalies/n_test)

        # shuffle
        if shuffle:
            # training set
            training_idxs = np.arange(n_training)
            np.random.shuffle(training_idxs)
            training_values = training_values[training_idxs]
            # test set
            test_idxs = np.arange(n_test)
            np.random.shuffle(test_idxs)
            test_values = test_values[test_idxs]
            test_labels = test_labels[test_idxs]

        # potentially reshape values if dealing with image data
        if self.classification_dataset.name in image_datasets:
            training_values = reshape_images(training_values,
                self.classification_dataset.name, flatten_images)
            test_values = reshape_images(test_values,
                self.classification_dataset.name, flatten_images)

        return (training_values, test_values, test_labels), description


    def sample_multiple(self, n_training: int, n_test: int, n_steps: int = 10,
               known_outlier_class: int = 1,   
               n_known_outlier_classes: int = 1,
               ratio_known_normal: float = 0.,
               ratio_known_outlier: float = 0.,
               ratio_pollution: float = 0.,
               training_contamination_rate: float = 0, #todo: verify if still necessary
               test_contamination_rate: float = 0.2, #todo: verify if still necessary
               shuffle: bool = True, random_seed: float = 42,
               apply_random_seed: bool =True,
               training_keep_frequency_ratio_normals: bool = False,
               training_equal_frequency_normals: bool = False,
               training_keep_frequency_ratio_anomalies: bool = False,
               training_equal_frequency_anomalies: bool = False,
               test_keep_frequency_ratio_normals: bool = False,
               test_equal_frequency_normals: bool = False,
               test_keep_frequency_ratio_anomalies: bool = False,
               test_equal_frequency_anomalies: bool = False,
               include_description: bool = True, yamlpath_append: Optional = None,
               yamlpath_new: Optional = None, flatten_images: bool = True, verbose: bool = False):
        """
        Sample multiple times from the anomaly dataset as an iterator.
        Note that this method does not ensure that data points seen during
        training are resampled during testing. If possible, use other methods
        like sample_with_training_split or sample_with_explicit_numbers
        that ensure this.

        :param n_training: Number of training data points to sample
        :param n_test: Number of test data points to sample
        :param n_steps: Number of sampled to take, i.e., number of times
            sampling is repeated, defaults to 10
        :param training_contamination_rate: Contamination rate when sampling
            training points, defaults to 0
        :param test_contamination_rate: Contamination rate when sampling
            test points, defaults to 0
        :param shuffle: Shuffle dataset, defaults to True
        :param random_seed: Random seed, defaults to 42
        :param apply_random_seed: Apply random seed to make sampling reproducible, defaults to True
        :param training_keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling for training. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points for training are sampled, 200 of them will be from
            label 0 and 100 from label 1. Defaults to False
        :param training_equal_frequency_normals: If there are multiple normal labels and
             this is set, training data will be sampled with an equal distribution
             among these normal labels, defaults to False
        :param training_keep_frequency_ratio_anomalies: Like parameter
            `training_keep_frequency_ratio_normals` for anomalies in training,
            defaults to False
        :param training_equal_frequency_anomalies: Like parameter
            `training_equal_frequency_normals` for anomalies in training,
            defaults to False
        :param test_keep_frequency_ratio_normals: Like parameter
            `training_keep_frequency_ratio_normals` for normals in test data,
            defaults to False
        :param test_equal_frequency_normals: Like parameter
            `training_equal_frequency_normals` for normals in test data,
            defaults to False
        :param test_keep_frequency_ratio_anomalies: Like parameter
            `training_keep_frequency_ratio_normals` for anomalies in test data,
            defaults to False
        :param test_equal_frequency_anomalies: Like parameter
            `training_equal_frequency_normals` for anomalies in test data,
            defaults to False
        :include_description: Includes sampling config file, defaults to True
        :param yamlpath_append: Optionally append sampling arguments to a YAML
            if this is not None, the path of the YAML is specified in this
            argument, defaults to None
        :param yamlpath_new: Optionally create a new YAML with the sampling
            arguments if this is not None, the path of the YAML is specified in
            this argument, defaults to None
        :param flatten_images: This flag has behavior only if an image dataset
            is loaded. If this is the case and this is set to True, the
            image will be flattened. If it is set to False, the original dimensionality
            of the image will be used, e.g., 32x32x3. Defaults to True

        :return: An iterator of A tuple (x_train, x_test, y_test), sample_config.
        """
        kwargs = {'n_training': n_training, 'n_test': n_test,
                  'training_contamination_rate': training_contamination_rate,
                  'training_keep_frequency_ratio_normals': training_keep_frequency_ratio_normals,
                  'training_equal_frequency_normals': training_equal_frequency_normals,
                  'training_keep_frequency_ratio_anomalies': training_keep_frequency_ratio_anomalies,
                  'training_equal_frequency_anomalies': training_equal_frequency_anomalies,
                  'test_contamination_rate': test_contamination_rate,
                  'test_keep_frequency_ratio_normals': test_keep_frequency_ratio_normals,
                  'test_equal_frequency_normals': test_equal_frequency_normals,
                  'test_keep_frequency_ratio_anomalies': test_keep_frequency_ratio_anomalies,
                  'test_equal_frequency_anomalies': test_equal_frequency_anomalies,
                  'shuffle': shuffle, 'random_seed': random_seed,
                  'apply_random_seed': apply_random_seed, 'include_description': include_description,
                  'flatten_images': flatten_images
        }
        # store arguments in YAML if specified
        if yamlpath_append: # append arguments to YAML
            _append_sampling_to_yaml(yamlpath_append, "semisupervised_multiple", kwargs)
        if yamlpath_new: # create new YAML with arguments
            _make_yaml(yamlpath_new, "sampling", {'semisupervised_multiple': kwargs})
        # sample with specified parameters first
        yield self.sample(**kwargs)
        for _ in range(1, n_steps):
            # increase random seed by 1 to make sure sampling is actually the
            # same, even when an algorithm also uses a random call somewhere
            kwargs['random_seed'] = kwargs['random_seed'] + 1
            yield self.sample(**kwargs)
            

    def sample_with_training_split(self, training_split: float,
        known_outlier_class: int = 1,   
        n_known_outlier_classes: int = 1,
        ratio_known_normal: float = 0.,
        ratio_known_outlier: float = 0.,
        ratio_pollution: float = 0.,
        max_contamination_rate: float = 0., #todo: verify if still necessary
        random_seed : float =42,
        apply_random_seed : bool = True,
        keep_frequency_ratio_normals: bool = False,
        equal_frequency_normals: bool = False,
        keep_frequency_ratio_anomalies: bool = False,
        equal_frequency_anomalies: bool = False,
        yamlpath_append: Optional = None,
        yamlpath_new: Optional = None, flatten_images: bool = True, verbose: bool =False):
        """
        Sample from a semisupervised anomaly dataset by specifying the split
        of normal data points used during training and a maximum contamination
        rate of the test set.

        :param training_split: Specifies the proportion of normal data points
            that will be used during training
        :param max_contamination_rate: Maximum contamination rate of the test
            set. If this is exceeded, not all anomalies that exist are sampled
        :param random_seed: Seed for random number generator, defaults to 42
        :param apply_random_seed: Whether or not to apply random seed, defaults to True
        :param keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points are sampled, 200 of them will be from
            label 0 and 100 from label 1. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param equal_frequency_normals: If there are multiple normal labels and
             this is set, data will be sampled with an equal distribution
             among these normal labels. Note that the split of these values
             between train and test set is at random. Defaults to False
        :param keep_frequency_ratio_anomalies: If there are multiple anomaly labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if anomaly labels are [2, 3] and there
            are 2000 data points with label 2, 1000 data points with label 3 and
            300 anomalous data points are sampled, 200 of them will be from
            label 2 and 100 from label 3. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param equal_frequency_anomalies: If there are multiple anomaly labels and
             this is set, data will be sampled with an equal distribution
            among these anomaly labels. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param yamlpath_append: Optionally append sampling arguments to a YAML
            if this is not None, the path of the YAML is specified in this
            argument, defaults to None
        :param yamlpath_new: Optionally create a new YAML with the sampling
            arguments if this is not None, the path of the YAML is specified in
            this argument, defaults to None
        :param flatten_images: This flag has behavior only if an image dataset
            is loaded. If this is the case and this is set to True, the
            image will be flattened. If it is set to False, the original dimensionality
            of the image will be used, e.g., 32x32x3. Defaults to True

        :return: A tuple of (training_values, test_values, test_labels), description
            where values and labels are a numpy.ndarray
        """        
        kwargs = {
            'training_split': training_split,
            'known_outlier_class': known_outlier_class,
            'n_known_outlier_classes': n_known_outlier_classes,
            'ratio_known_normal': ratio_known_normal,
            'ratio_known_outlier': ratio_known_outlier,
            'ratio_pollution': ratio_pollution,
            'max_contamination_rate': max_contamination_rate,
            'random_seed': random_seed, 'apply_random_seed': apply_random_seed,
            'keep_frequency_ratio_normals': keep_frequency_ratio_normals,
            'equal_frequency_normals': equal_frequency_normals,
            'keep_frequency_ratio_anomalies': keep_frequency_ratio_anomalies,
            'equal_frequency_anomalies': equal_frequency_anomalies,
            'flatten_images': flatten_images,
            'verbose': verbose
        }
        # store arguments in YAML if specified
        if yamlpath_append: # append arguments to YAML
            _append_sampling_to_yaml(yamlpath_append, "gsemisupervised_training_split_single", kwargs)
        if yamlpath_new: # create new YAML with arguments
            _make_yaml(yamlpath_new, "sampling", {'gsemisupervised_training_split_single': kwargs})

        normal_class = self.normal_labels # Specify the normal class of the dataset (all other classes are considered anomalous).

        outlier_classes = self.anomaly_labels
        outlier_classes = list(set(outlier_classes)-set(normal_class))

        normal_classes = tuple([normal_class])
        outlier_classes = tuple([outlier_classes]) # 1-9
        
        self.known_outlier_class = known_outlier_class
        self.n_known_outlier_classes = n_known_outlier_classes        
        self.ratio_known_normal = ratio_known_normal
        self.ratio_known_outlier = ratio_known_outlier
        self.ratio_pollution = ratio_pollution
        
        seed=(random_seed,42)[apply_random_seed]
        
        X = self.classification_dataset.values
        y = self.classification_dataset.labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.-training_split, shuffle=True, random_state=seed)
        
        if verbose:
            print(f"trn: {X_train.shape} {y_train.shape}")
            print(f"tst: {X_test.shape} {y_test.shape}")
        
        list_idx_trn, list_labels_trn, list_semi_labels_trn = self._create_semisupervised_setting(        
            y_train,
            training_split,
            normal_classes, # Specify the normal class of the dataset (all other classes are considered anomalous).
            outlier_classes,
            known_outlier_class, #def 1 # Specify the known outlier class of the dataset for semi-supervised anomaly detection.
            n_known_outlier_classes, #def 0  # Number of known outlier classes.'
                                             # 'If 0, no anomalies are known.'
                                             # 'If 1, outlier class as specified in --known_outlier_class option.'
                                             # 'If > 1, the specified number of outlier classes will be sampled at random.'
            ratio_known_normal, #def 0.0   # Ratio of known (labeled) normal training examples.
            ratio_known_outlier, #def 0 # Ratio of known (labeled) anomalous training example
            ratio_pollution, #def 0.0    # Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.
            verbose=verbose)
                
        self.classification_dataset.labels_semi_trn[list_idx_trn] = torch.tensor(list_semi_labels_trn).numpy()  # set semi-supervised labels
        
        x_trn = X_train[list_idx_trn]
        y_trn_semi = np.array(list_semi_labels_trn)
        x_tst = X_test
        y_tst = y_test
        
        n_normal_trn = len(np.argwhere(np.isin(list_labels_trn, normal_classes)).flatten())
        n_normal_tst = len(np.argwhere(np.isin(y_test, normal_classes)).flatten())
        n_outlier_trn = len(np.argwhere(np.isin(list_labels_trn, outlier_classes)).flatten())
        n_outlier_tst = len(np.argwhere(np.isin(y_test, outlier_classes)).flatten())
                                            
        n_trn = n_normal_trn + n_outlier_trn
        n_tst = n_normal_tst + n_outlier_tst
        
        # data description
        description = GSemisupervisedAnomalyDatasetDescription(name=self.classification_dataset.name,
            normal_labels=self.normal_labels, normal_classes=self.normal_classes, 
            anomaly_labels=self.anomaly_labels, outlier_classes=self.outlier_classes,
            known_outlier_class=self.known_outlier_class, n_known_outlier_classes=self.n_known_outlier_classes,
            ratio_known_normal=self.ratio_known_normal, ratio_known_outlier=self.ratio_known_outlier, ratio_pollution=self.ratio_pollution,
            number_instances_training=n_trn,
            number_instances_test=n_tst,
            training_number_normals=n_normal_trn,
            training_number_anomalies=n_outlier_trn,
            training_contamination_rate=n_outlier_trn/n_trn,
            test_number_normals=n_normal_tst,
            test_number_anomalies=n_outlier_tst,
            test_contamination_rate=n_outlier_tst/n_tst)
                
        return (x_trn, y_trn_semi, x_tst, y_tst), description
    

    def sample_multiple_with_training_split(self, training_split: float,
        known_outlier_class: int = 1,   
        n_known_outlier_classes: int = 1,
        ratio_known_normal: float = 0.,
        ratio_known_outlier: float = 0.,
        ratio_pollution: float = 0.,
        max_contamination_rate: float = 0., #todo: verify if still necessary
        n_steps: int =10, random_seed : float =42,
        apply_random_seed : bool = True,
        keep_frequency_ratio_normals: bool = False,
        equal_frequency_normals: bool = False,
        keep_frequency_ratio_anomalies: bool = False,
        equal_frequency_anomalies: bool = False,
        yamlpath_append: Optional = None,
        yamlpath_new: Optional = None, flatten_images: bool = True, verbose: bool = False):
        """
        ratio_known_normal = 0. #def 0.0  # Ratio of known (labeled) normal training examples.
        ratio_known_outlier = 0.01 #def 0 # Ratio of known (labeled) anomalous training example
        ratio_pollution = 0.1 #def 0.0    # Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.                
        known_outlier_class=1 #def 1      # Specify the known outlier class of the dataset for semi-supervised anomaly detection.
        n_known_outlier_classes=1 #def 0  # Number of known outlier classes.'
                                          # 'If 0, no anomalies are known.'
                                          # 'If 1, outlier class as specified in --known_outlier_class option.'
                                          # 'If > 1, the specified number of outlier classes will be sampled at random.'
        """
        """
        Sample multiple times from a semisupervised anomaly dataset by specifying the split
        of normal data points used during training and a maximum contamination
        rate of the test set.

        :param training_split: Specifies the proportion of normal data points
            that will be used during training
        :param max_contamination_rate: Maximum contamination rate of the test
            set. If this is exceeded, not all anomalies that exist are sampled
        :param n_steps: Number of samples to be taken
        :param random_seed: Seed for random number generator in the first sample,
            defaults to 42
        :param apply_random_seed: Whether or not to apply random seed, defaults to True
        :param keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points are sampled, 200 of them will be from
            label 0 and 100 from label 1. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param equal_frequency_normals: If there are multiple normal labels and
             this is set, data will be sampled with an equal distribution
             among these normal labels. Note that the split of these values
             between train and test set is at random. Defaults to False
        :param keep_frequency_ratio_anomalies: If there are multiple anomaly labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if anomaly labels are [2, 3] and there
            are 2000 data points with label 2, 1000 data points with label 3 and
            300 anomalous data points are sampled, 200 of them will be from
            label 2 and 100 from label 3. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param equal_frequency_anomalies: If there are multiple anomaly labels and
             this is set, data will be sampled with an equal distribution
            among these anomaly labels. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param yamlpath_append: Optionally append sampling arguments to a YAML
            if this is not None, the path of the YAML is specified in this
            argument, defaults to None
        :param yamlpath_new: Optionally create a new YAML with the sampling
            arguments if this is not None, the path of the YAML is specified in
            this argument, defaults to None
        :param flatten_images: This flag has behavior only if an image dataset
            is loaded. If this is the case and this is set to True, the
            image will be flattened. If it is set to False, the original dimensionality
            of the image will be used, e.g., 32x32x3. Defaults to True

        :return: An iterator of tuples of (training_values, test_values, test_labels), description
            where values and labels are a numpy.ndarray
        """
        
        kwargs = {'training_split': training_split,
            'known_outlier_class': known_outlier_class,
            'n_known_outlier_classes': n_known_outlier_classes,
            'ratio_known_normal': ratio_known_normal, 
            'ratio_known_outlier': ratio_known_outlier, 
            'ratio_pollution': ratio_pollution,
            'max_contamination_rate': max_contamination_rate, 'n_steps': n_steps,
            'random_seed': random_seed, 'apply_random_seed': apply_random_seed,
            'keep_frequency_ratio_normals': keep_frequency_ratio_normals,
            'equal_frequency_normals': equal_frequency_normals,
            'keep_frequency_ratio_anomalies': keep_frequency_ratio_anomalies,
            'equal_frequency_anomalies': equal_frequency_anomalies,
            'flatten_images': flatten_images,
            'verbose': verbose
        }
        # store arguments in YAML if specified
        if yamlpath_append: # append arguments to YAML
            _append_sampling_to_yaml(yamlpath_append, "semisupervised_training_split_multiple", kwargs)
        if yamlpath_new: # create new YAML with arguments
            _make_yaml(yamlpath_new, "sampling", {'semisupervised_training_split_multiple': kwargs})

        del kwargs['n_steps']
        yield self.sample_with_training_split(**kwargs)
        
        for _ in range(1, n_steps):
            kwargs['random_seed'] = kwargs['random_seed'] + 1
            yield self.sample_with_training_split(**kwargs)

    def sample_original_mvtec_split(self, flatten_images: bool = True, verbose: bool = False):
        """
        Samples with the train-test split from the original publication of
        MVTec AD. This is included to allow reproducing experiments that follow
        this original train-test split.

        :param flatten_images: This flag has behavior only if an image dataset
            is loaded. If this is the case and this is set to True, the
            image will be flattened. If it is set to False, the original dimensionality
            of the image will be used, e.g., 32x32x3. Defaults to True

        :return: A tuple (x_train, x_test, y_test), sample_config. y_test and
            sample_config can be used to pass to the EvaluationObject.
        """
        # only mvtec_ad datasets store which splits were originally used
        # in the dataset publication
        if not hasattr(self.classification_dataset, 'x_normal_test_mask'):
            raise Error(f"This dataset is not loaded as MVTec AD dataset "\
                f"and therefore doesn't provide this functionality.")

        x_normal_test_mask = self.classification_dataset.x_normal_test_mask.astype('bool')
        normals = len(x_normal_test_mask)
        x_normal_train_mask = ~x_normal_test_mask
        x_train = self.classification_dataset.values[:normals][x_normal_train_mask]
        x_test_normals = self.classification_dataset.values[:normals][x_normal_test_mask]
        x_test_anomalies = self.classification_dataset.values[normals:]
        x_test = np.vstack((x_test_normals, x_test_anomalies))
        y = np.hstack((np.zeros(len(x_test_normals)), np.ones(len(x_test_anomalies))))

        sampling_config = GSemisupervisedAnomalyDatasetDescription(
            name=self.classification_dataset.name,
            normal_labels=[0], anomaly_labels=[1],
            number_instances_training=len(x_train),
            number_instances_test=len(x_test),
            training_number_normals=len(x_train),
            training_number_anomalies=0,
            training_contamination_rate=0,
            test_number_normals=len(x_test_normals),
            test_number_anomalies=len(x_test_anomalies),
            test_contamination_rate=len(x_test_anomalies)/len(x_test),
        )

        # reshape images
        x_train = reshape_images(x_train,
            dataset_name=self.classification_dataset.name,
            flatten_images=flatten_images)
        x_test = reshape_images(x_test,
            dataset_name=self.classification_dataset.name,
            flatten_images=flatten_images)

        return (x_train, x_test, y), sampling_config

    def _create_semisupervised_setting(self, labels,
        training_split, 
        normal_classes, # Specify the normal class of the dataset (all other classes are considered anomalous).
        outlier_classes,
        known_outlier_class, #def 1 # Specify the known outlier class of the dataset for semi-supervised anomaly detection.
        n_known_outlier_classes, #def 0  # Number of known outlier classes.'
                                         # 'If 0, no anomalies are known.'
                                         # 'If 1, outlier class as specified in --known_outlier_class option.'
                                         # 'If > 1, the specified number of outlier classes will be sampled at random.'
        ratio_known_normal, #def 0.0   # Ratio of known (labeled) normal training examples.
        ratio_known_outlier, #def 0 # Ratio of known (labeled) anomalous training example
        ratio_pollution, #def 0.0    # Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.
        verbose=False): 
                
        if n_known_outlier_classes == 0:
            known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            known_outlier_classes = tuple([known_outlier_class])
        else:
            known_outlier_classes = tuple(random.sample(outlier_classes, n_known_outlier_classes)) # sample n from range 1-9
                        
        if verbose:
            print(f"---------------------------------------------")
            print(f"normal_classes {normal_classes}")
            print(f"outlier_classes {outlier_classes}")
            print(f"known_outlier_classes {known_outlier_classes}")
            print(f"ratio_known_normal {ratio_known_normal}")
            print(f"ratio_known_outlier {ratio_known_outlier}")
            print(f"ratio_pollution {ratio_pollution}")

        idx_normal = np.argwhere(np.isin(labels, normal_classes)).flatten()
        idx_outlier = np.argwhere(np.isin(labels, outlier_classes)).flatten()
        idx_known_outlier_candidates = np.argwhere(np.isin(labels, known_outlier_classes)).flatten()

        if verbose:
            print(f"---------------------------------------------")
            print(f"labels {labels} ({len(labels)})")
            print(f"normal_classes {normal_classes}")
            print(f"idx_normal: {len(idx_normal)} -> {idx_normal}")
            print(f"outlier_classes {outlier_classes}")
            print(f"idx_outlier: {len(idx_outlier)} -> {idx_outlier}")
            print(f"known_outlier_classes {known_outlier_classes}")
            print(f"idx_known_outlier: {len(idx_known_outlier_candidates)} -> {idx_known_outlier_candidates}")
            print(f"ratio_known_normal {ratio_known_normal}")
            print(f"ratio_known_outlier {ratio_known_outlier}")
            print(f"ratio_pollution {ratio_pollution}")
            print(f"total: {len(idx_normal)+len(idx_outlier)}")

        n_normal = len(idx_normal)

        # Solve system of linear equations to obtain respective number of samples
        a = np.array([[1, 1, 0, 0],
                      [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                      [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                      [0, -ratio_pollution, (1-ratio_pollution), 0]])
        b = np.array([n_normal, 0, 0, 0])
        x = np.linalg.solve(a, b)

        # Get number of samples
        n_known_normal = int(x[0])
        n_unlabeled_normal = int(x[1])
        n_unlabeled_outlier = int(x[2])
        n_known_outlier = int(x[3])

        if verbose:
            print(f"---------------------------------------------solve")
            print(f"n_known_normal: {n_known_normal}")
            print(f"n_unlabeled_normal: {n_unlabeled_normal}")
            print(f"n_unlabeled_outlier: {n_unlabeled_outlier}")
            print(f"n_known_outlier: {n_known_outlier}")
            print(f"n_total: {n_known_normal+n_unlabeled_normal+n_unlabeled_outlier+n_known_outlier}")

        # Sample indices
        perm_normal = np.random.permutation(n_normal)
        perm_outlier = np.random.permutation(len(idx_outlier))
        perm_known_outlier = np.random.permutation(len(idx_known_outlier_candidates))

        if verbose:
            print(f"---------------------------------------------")
            print(f"labels {labels} ({len(labels)})")
            print(f"idx_normal {idx_normal}")
            print(f"perm_normal {perm_normal}")
            print(f"idx_outlier {idx_outlier}")
            print(f"perm_outlier {perm_outlier}")

        idx_known_normal = idx_normal[perm_normal[:n_known_normal]].tolist()
        idx_unlabeled_normal = idx_normal[perm_normal[n_known_normal:n_known_normal+n_unlabeled_normal]].tolist()
        idx_unlabeled_outlier = idx_outlier[perm_outlier[:n_unlabeled_outlier]].tolist()
        idx_known_outlier = idx_known_outlier_candidates[perm_known_outlier[:n_known_outlier]].tolist()

        if verbose:
            print(f"---------------------------------------------")
            print(f"labels {labels} ({len(labels)})")
            print(f"idx_normal {idx_normal}")
            print(f"idx_outlier {idx_outlier}")
            print(f"known_normal: {n_known_normal} -> {idx_known_normal}")
            print(f"unlabeled_normal: {n_unlabeled_normal} -> {idx_unlabeled_normal}")
            print(f"unlabeled_outlier: {n_unlabeled_outlier} -> {idx_unlabeled_outlier}")
            print(f"known_outlier: {n_known_outlier} -> {idx_known_outlier}")
            print(f"total: {n_known_normal+n_unlabeled_normal+n_unlabeled_outlier+n_known_outlier}")

        # Get original class labels
        labels_known_normal = labels[idx_known_normal].tolist()
        labels_unlabeled_normal = labels[idx_unlabeled_normal].tolist()
        labels_unlabeled_outlier = labels[idx_unlabeled_outlier].tolist()
        labels_known_outlier = labels[idx_known_outlier].tolist()

        if verbose:
            print(f"---------------------------------------------")
            print(f"labels {labels} ({len(labels)})")
            print(f"idx_normal {idx_normal}")
            print(f"idx_outlier {idx_outlier}")
            print(f"known_normal: {n_known_normal}")
            print(f"known_normal: {len(idx_known_normal)} -> {idx_known_normal}")
            print(f"unlabeled_normal: {n_unlabeled_normal}")
            print(f"unlabeled_normal: {len(idx_unlabeled_normal)} -> {idx_unlabeled_normal}")
            print(f"unlabeled_outlier: {n_unlabeled_outlier}")
            print(f"unlabeled_outlier: {len(idx_unlabeled_outlier)} -> {idx_unlabeled_outlier}")
            print(f"known_outlier: {n_known_outlier}")
            print(f"known_outlier: {len(idx_known_outlier)} -> {idx_known_outlier}")
            print(f"total normal: {len(idx_known_normal)+len(idx_unlabeled_normal)}")
            print(f"total outlier: {len(idx_unlabeled_outlier)+len(idx_known_outlier)}")
            print(f"total: {len(idx_known_normal)+len(idx_unlabeled_normal)+len(idx_unlabeled_outlier)+len(idx_known_outlier)}")

        # Get semi-supervised setting labels
        semi_labels_known_normal = np.ones(n_known_normal).astype(np.int32).tolist()
        semi_labels_unlabeled_normal = np.zeros(n_unlabeled_normal).astype(np.int32).tolist()
        semi_labels_unlabeled_outlier = np.zeros(n_unlabeled_outlier).astype(np.int32).tolist()
        semi_labels_known_outlier = (-np.ones(n_known_outlier).astype(np.int32)).tolist()

        # Create final lists
        list_idx = idx_known_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_known_outlier
        list_labels = labels_known_normal + labels_unlabeled_normal + labels_unlabeled_outlier + labels_known_outlier
        list_semi_labels = (semi_labels_known_normal + semi_labels_unlabeled_normal + semi_labels_unlabeled_outlier + semi_labels_known_outlier)

        return list_idx, list_labels, list_semi_labels

    

    def _create_semisupervised_setting2(self, labels,
        training_split, 
        normal_classes, # Specify the normal class of the dataset (all other classes are considered anomalous).
        outlier_classes,
        known_outlier_class, #def 1 # Specify the known outlier class of the dataset for semi-supervised anomaly detection.
        n_known_outlier_classes, #def 0  # Number of known outlier classes.'
                                         # 'If 0, no anomalies are known.'
                                         # 'If 1, outlier class as specified in --known_outlier_class option.'
                                         # 'If > 1, the specified number of outlier classes will be sampled at random.'
        ratio_known_normal, #def 0.0   # Ratio of known (labeled) normal training examples.
        ratio_known_outlier, #def 0 # Ratio of known (labeled) anomalous training example
        ratio_pollution, #def 0.0    # Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.
        verbose=False): 
                
        if n_known_outlier_classes == 0:
            known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            known_outlier_classes = tuple([known_outlier_class])
        else:
            known_outlier_classes = tuple(random.sample(outlier_classes, n_known_outlier_classes)) # sample n from range 1-9
                        
        if verbose:
            print(f"---------------------------------------------")
            print(f"normal_classes {normal_classes}")
            print(f"outlier_classes {outlier_classes}")
            print(f"known_outlier_classes {known_outlier_classes}")
            print(f"ratio_known_normal {ratio_known_normal}")
            print(f"ratio_known_outlier {ratio_known_outlier}")
            print(f"ratio_pollution {ratio_pollution}")

        idx_normal = np.argwhere(np.isin(labels, normal_classes)).flatten()
        idx_outlier = np.argwhere(np.isin(labels, outlier_classes)).flatten()
        idx_known_outlier_candidates = np.argwhere(np.isin(labels, known_outlier_classes)).flatten()
          
        if verbose:
            print(f"---------------------------------------------")
            print(f"labels {labels} ({len(labels)})")
            print(f"normal_classes {normal_classes}")
            print(f"idx_normal: {len(idx_normal)} -> {idx_normal}")
            print(f"outlier_classes {outlier_classes}")
            print(f"idx_outlier: {len(idx_outlier)} -> {idx_outlier}")
            print(f"known_outlier_classes {known_outlier_classes}")
            print(f"idx_known_outlier: {len(idx_known_outlier_candidates)} -> {idx_known_outlier_candidates}")
            print(f"ratio_known_normal {ratio_known_normal}")
            print(f"ratio_known_outlier {ratio_known_outlier}")
            print(f"ratio_pollution {ratio_pollution}")
            print(f"total: {len(idx_normal)+len(idx_outlier)}")

        n_normal = len(idx_normal)

        # Solve system of linear equations to obtain respective number of samples
        a = np.array([[1, 1, 0, 0],
                      [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                      [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                      [0, -ratio_pollution, (1-ratio_pollution), 0]])
        b = np.array([n_normal, 0, 0, 0])
        x = np.linalg.solve(a, b)

        n_known_normal = int(x[0])
        n_unlabeled_normal = int(x[1])
        n_unlabeled_outlier = int(x[2])
        n_known_outlier = int(x[3])

        if verbose:
            print(f"---------------------------------------------solve")
            print(f"n_known_normal: {n_known_normal}")
            print(f"n_unlabeled_normal: {n_unlabeled_normal}")
            print(f"n_unlabeled_outlier: {n_unlabeled_outlier}")
            print(f"n_known_outlier: {n_known_outlier}")
            print(f"n_total: {n_known_normal+n_unlabeled_normal+n_unlabeled_outlier+n_known_outlier}")

        # Sample indices
        perm_normal = np.random.permutation(n_normal)
        perm_outlier = np.random.permutation(len(idx_outlier))
        perm_known_outlier = np.random.permutation(len(idx_known_outlier_candidates))

        if verbose:
            print(f"---------------------------------------------")
            print(f"labels {labels} ({len(labels)})")
            print(f"idx_normal {idx_normal}")
            print(f"perm_normal {perm_normal}")
            print(f"idx_outlier {idx_outlier}")
            print(f"perm_outlier {perm_outlier}")

        idx_known_normal = idx_normal[perm_normal[:n_known_normal]].tolist()
        idx_unlabeled_normal = idx_normal[perm_normal[n_known_normal:n_known_normal+n_unlabeled_normal]].tolist()
        idx_unlabeled_outlier = idx_outlier[perm_outlier[:n_unlabeled_outlier]].tolist()
        idx_known_outlier = idx_known_outlier_candidates[perm_known_outlier[:n_known_outlier]].tolist()

        if verbose:
            print(f"---------------------------------------------")
            print(f"labels {labels} ({len(labels)})")
            print(f"idx_normal {idx_normal}")
            print(f"idx_outlier {idx_outlier}")
            print(f"known_normal: {n_known_normal} -> {idx_known_normal}")
            print(f"unlabeled_normal: {n_unlabeled_normal} -> {idx_unlabeled_normal}")
            print(f"unlabeled_outlier: {n_unlabeled_outlier} -> {idx_unlabeled_outlier}")
            print(f"known_outlier: {n_known_outlier} -> {idx_known_outlier}")
            print(f"total: {n_known_normal+n_unlabeled_normal+n_unlabeled_outlier+n_known_outlier}")
        
        # Get original class labels
        labels_known_normal = labels[idx_known_normal].tolist()
        labels_unlabeled_normal = labels[idx_unlabeled_normal].tolist()
        labels_unlabeled_outlier = labels[idx_unlabeled_outlier].tolist()
        labels_known_outlier = labels[idx_known_outlier].tolist()

        idx_known_normal_trn, idx_known_normal_tst = train_test_split_(training_split, idx_known_normal, lst=True)
        idx_unlabeled_normal_trn, idx_unlabeled_normal_tst = train_test_split_(training_split, idx_unlabeled_normal, lst=True)
        idx_unlabeled_outlier_trn, idx_unlabeled_outlier_tst = train_test_split_(training_split, idx_unlabeled_outlier, lst=True)
        idx_known_outlier_trn, idx_known_outlier_tst = train_test_split_(training_split, idx_known_outlier, lst=True)

        if verbose:
            print(f"---------------------------------------------")
            print(f"labels {labels} ({len(labels)})")
            print(f"idx_normal {idx_normal}")
            print(f"idx_outlier {idx_outlier}")
            print(f"known_normal: {n_known_normal}")
            print(f"known_normal_trn: {len(idx_known_normal_trn)} -> {idx_known_normal_trn}")
            print(f"known_normal_tst: {len(idx_known_normal_tst)} -> {idx_known_normal_tst}")
            print(f"unlabeled_normal: {n_unlabeled_normal}")
            print(f"unlabeled_normal_trn: {len(idx_unlabeled_normal_trn)} -> {idx_unlabeled_normal_trn}")
            print(f"unlabeled_normal_tst: {len(idx_unlabeled_normal_tst)} -> {idx_unlabeled_normal_tst}")
            print(f"unlabeled_outlier: {n_unlabeled_outlier}")
            print(f"unlabeled_outlier_trn: {len(idx_unlabeled_outlier_trn)} -> {idx_unlabeled_outlier_trn}")
            print(f"unlabeled_outlier_tst: {len(idx_unlabeled_outlier_tst)} -> {idx_unlabeled_outlier_tst}")
            print(f"known_outlier: {n_known_outlier}")
            print(f"known_outlier_trn: {len(idx_known_outlier_trn)} -> {idx_known_outlier_trn}")
            print(f"known_outlier_tst: {len(idx_known_outlier_tst)} -> {idx_known_outlier_tst}")
            print(f"total trn normal: {len(idx_known_normal_trn)+len(idx_unlabeled_normal_trn)}")
            print(f"total tst normal: {len(idx_known_normal_tst)+len(idx_unlabeled_normal_tst)}")
            print(f"total trn outlier: {len(idx_unlabeled_outlier_trn)+len(idx_known_outlier_trn)}")
            print(f"total tst outlier: {len(idx_unlabeled_outlier_tst)+len(idx_known_outlier_tst)}")
            print(f"total trn: {len(idx_known_normal_trn)+len(idx_unlabeled_normal_trn)+len(idx_unlabeled_outlier_trn)+len(idx_known_outlier_trn)}")
            print(f"total tst: {len(idx_known_normal_tst)+len(idx_unlabeled_normal_tst)+len(idx_unlabeled_outlier_tst)+len(idx_known_outlier_tst)}")
            #print(f"total: {n_known_normal+n_unlabeled_normal+n_unlabeled_outlier+n_known_outlier}")
            print(f"total: {len(idx_known_normal_trn)+len(idx_unlabeled_normal_trn)+len(idx_unlabeled_outlier_trn)+len(idx_known_outlier_trn)+len(idx_known_normal_tst)+len(idx_unlabeled_normal_tst)+len(idx_unlabeled_outlier_tst)+len(idx_known_outlier_tst)}")

        labels_known_normal_trn, labels_known_normal_tst = labels[idx_known_normal_trn].tolist(), labels[idx_known_normal_tst].tolist()
        labels_unlabeled_normal_trn, labels_unlabeled_normal_tst = labels[idx_unlabeled_normal_trn].tolist(), labels[idx_unlabeled_normal_tst].tolist()
        labels_unlabeled_outlier_trn, labels_unlabeled_outlier_tst = labels[idx_unlabeled_outlier_trn].tolist(), labels[idx_unlabeled_outlier_tst].tolist()
        labels_known_outlier_trn, labels_known_outlier_tst = labels[idx_known_outlier_trn].tolist(), labels[idx_known_outlier_tst].tolist()

        # Get semi-supervised setting labels
        semi_labels_known_normal = np.ones(n_known_normal).astype(np.int32).tolist()
        semi_labels_unlabeled_normal = np.zeros(n_unlabeled_normal).astype(np.int32).tolist()
        semi_labels_unlabeled_outlier = np.zeros(n_unlabeled_outlier).astype(np.int32).tolist()
        semi_labels_known_outlier = (-np.ones(n_known_outlier).astype(np.int32)).tolist()

        semi_labels_known_normal_trn, semi_labels_known_normal_tst = train_test_split_(training_split, semi_labels_known_normal, lst=True)
        semi_labels_unlabeled_normal_trn, semi_labels_unlabeled_normal_tst = train_test_split_(training_split, semi_labels_unlabeled_normal, lst=True)
        semi_labels_unlabeled_outlier_trn, semi_labels_unlabeled_outlier_tst = train_test_split_(training_split, semi_labels_unlabeled_outlier, lst=True)
        semi_labels_known_outlier_trn, semi_labels_known_outlier_tst = train_test_split_(training_split, semi_labels_known_outlier, lst=True)

        # Create final lists
        list_idx_trn = idx_known_normal_trn + idx_unlabeled_normal_trn + idx_unlabeled_outlier_trn + idx_known_outlier_trn
        list_labels_trn = labels_known_normal_trn + labels_unlabeled_normal_trn + labels_unlabeled_outlier_trn + labels_known_outlier_trn
        list_semi_labels_trn = (semi_labels_known_normal_trn + semi_labels_unlabeled_normal_trn + semi_labels_unlabeled_outlier_trn + semi_labels_known_outlier_trn)

        list_idx_tst = idx_known_normal_tst + idx_unlabeled_normal_tst + idx_unlabeled_outlier_tst + idx_known_outlier_tst
        list_labels_tst = labels_known_normal_tst + labels_unlabeled_normal_tst + labels_unlabeled_outlier_tst + labels_known_outlier_tst
        list_semi_labels_tst = (semi_labels_known_normal_tst + semi_labels_unlabeled_normal_tst + semi_labels_unlabeled_outlier_tst + semi_labels_known_outlier_tst)

        if verbose:
            print(f"---------------------------------------------")
            print(f"list_idx_trn: {len(list_idx_trn)}")
            print(f"list_labels_trn: {len(list_labels_trn)}")
            print(f"list_semi_labels_trn: {len(list_semi_labels_trn)}")
            print(f"list_idx_tst: {len(list_idx_tst)}")
            print(f"list_labels_tst: {len(list_labels_tst)}")
            print(f"total: {len(list_idx_trn)+len(list_idx_tst)}")
            print(f"---------------------------------------------")
            
        return list_idx_trn, list_labels_trn, list_semi_labels_trn, list_idx_tst, list_labels_tst
        
        

    def _compute_samples_in_training_and_testing(self, training_split: float, verbose=False) -> Tuple[int, int, int, int]:
        """
        Helper that computes how many normal data points are needed for training
        data and how many normal and anomalous data points are needed for
        testing with the specified parameters of training split and maximum
        contamination rate.
        """
        
        import math
        
        idx_normal = self.normal_idxs
        idx_outlier = self.anomaly_idxs

        n_normal = len(idx_normal)
        n_normals = len(self.normal_idxs)
        
        n_outlier = len(idx_outlier)
        n_anomalies = len(self.anomaly_idxs)
        
        n_normal_trn = int(math.floor(training_split * n_normal))
        n_normal_tst = n_normal - n_normal_trn 

        n_outlier_trn = int(math.floor(training_split  * n_outlier))
        n_outlier_tst = n_outlier - n_outlier_trn 

        if verbose:
            print(f"n_normal_trn: {n_normal_trn}")
            print(f"n_normal_tst: {n_normal_tst}")
            print(f"n_outlier_trn: {n_outlier_trn}")
            print(f"n_outlier_tst: {n_outlier_tst}")
            print(f"n_total: {n_normal_trn+n_outlier_trn+n_normal_tst+n_outlier_tst}")
        
        return (n_normal_trn, n_normal_tst, n_outlier_trn, n_outlier_tst)
