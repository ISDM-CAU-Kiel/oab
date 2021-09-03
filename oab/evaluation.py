import math
import numpy as np
import pandas as pd

from dataclasses import dataclass
from collections.abc import Iterable
from typing import List, Optional, Dict, Tuple
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from oab.metrics import (roc_auc_score, precision_n_score, adjusted_precision_n_score,
    average_precision_score, adjusted_average_precision_score, precision_recall_auc_score)
from oab.data.abstract_classes import AnomalyDatasetDescription
from oab.data.semisupervised import SemisupervisedAnomalyDatasetDescription


all_metrics = ['roc_auc', 'average_precision', 'adjusted_average_precision',
               'precision_n', 'adjusted_precision_n', 'precision_recall_auc']

metrics_dict = {
    'roc_auc': roc_auc_score,
    'average_precision': average_precision_score,
    'adjusted_average_precision': adjusted_average_precision_score,
    'precision_n': precision_n_score,
    'adjusted_precision_n': adjusted_precision_n_score,
    'precision_recall_auc': precision_recall_auc_score,
}


@dataclass
class EvaluationDescription():
    """This object is returned when evaluating an algorithm's performance
    on a single sample from a dataset. It includes the description of the
    dataset as well as the values for the defined metrics.
    """
    dataset_description: AnomalyDatasetDescription
    algorithm_name: Optional[str]
    metrics: dict()


    def pretty_print(self):
        """Pretty printer.
        """
        print(self.dataset_description.print_for_eval_intro())
        print(self.dataset_description.print_for_eval_specifics())
        print(f"  \t Metric")
        for metric in self.metrics.keys():
            print(f"{self.metrics[metric] :.3f} \t {metric}")



@dataclass
class EvaluationMultipleSamplesDescription():
    """This object is returned when evaluating an algorithm's performance
    on multiple samples from a single dataset. It includes the description of the
    dataset as well as the values for the defined metrics (means and standard
    deviation).
    """
    dataset_description: AnomalyDatasetDescription
    algorithm_name: Optional[str]
    means: dict()
    std_devs: dict()
    n_steps: int


    def pretty_print(self):
        """ Pretty printer.
        """
        print(self.dataset_description.print_for_eval_intro())
        print(f"Total of {self.n_steps} datasets. Per dataset:")
        print(self.dataset_description.print_for_eval_specifics())
        print(f"Mean \t Std_dev \t Metric")
        for metric in self.means.keys():
            print(f"{self.means[metric] :.3f} \t {self.std_devs[metric] :.3f} \t\t {metric}")



def evaluate(y: np.ndarray, y_pred_scores: np.ndarray,
             description: AnomalyDatasetDescription, print: bool = True,
             metrics: List[str] = ['roc_auc', 'average_precision', 'adjusted_average_precision'],
             algorithm_name: Optional[str] = None) -> EvaluationDescription:
    """ Evaluate an algorithm's performance and return evaluation object.

    :param y: Binary ground truth values (0: normals, 1: anomalies)
    :param y_pred_scores: The raw anomaly scores as returned by a fitted model
    :param description: Dataset description object
    :param print: Print the result of the evaluation, defaults to True
    :param metrics: List of strings with names of metrics that are to be
        calculated. All available metrics are listed at `all_metrics`, defaults to
        `['roc_auc', 'average_precision', 'adjusted_average_precision']`

    :return: EvaluationDescription object that contains information about the
        dataset and its performance
    """
    if description == None:
        raise ValueError(f"Description object missing")

    evaluation_description = EvaluationDescription(
        dataset_description=description, algorithm_name=algorithm_name,
        metrics=dict())
    for metric in metrics:
        evaluation_description.metrics[metric] = _evaluate_metric(metric, y, y_pred_scores)
    return evaluation_description



def evaluate_on_multiple_samples(ys: Iterable,
    y_pred_scoress: Iterable,
    descriptions: Iterable,
    print: bool = True,
    metrics: List[str] = ['roc_auc', 'average_precision', 'adjusted_average_precision'],
    algorithm_name: Optional = None,
    ignore_sampling_configs: bool = False) -> EvaluationMultipleSamplesDescription:
    """ Evaluate an algorithm's performance on multiple samples of the same dataset
    and return evaluation object.

    :param ys: Iterable (iterating through the samples) of binary ground truth
        values (0: normals, 1: anomalies)
    :param y_pred_scoress: Iterable (iterating through the samples) of the raw
        anomaly scores as returned by a fitted model
    :param descriptions: Iterable (iterating through the samples) of tataset
        description object
    :param print: Print the result of the evaluation, defaults to True
    :param metrics: List of strings with names of metrics that are to be
        calculated. All available metrics are listed at `all_metrics`, defaults to
        `['roc_auc', 'average_precision', 'adjusted_average_precision']
    :param ignore_sampling_configs: If this is set, the sampling configs
        are not checked for consistency, i.e., it is possible that different
        normal and anomaly labels, different sampling sizes, etc. are used.
        This can be used if multiple different datasets should be considered
        as one, e.g., in the case of MVTec AD. Defaults to False

    :return: EvaluationMultipleSamplesDescription object that contains
        information about the dataset and its performance
    """
    # test that first three inputs are iterable
    if not isinstance(descriptions, Iterable):
        raise ValueError(f"descriptions must be iterable.")
    if (not isinstance(ys, Iterable)) or (not isinstance(y_pred_scoress, Iterable)):
        raise ValueError(f"Ground truth and predictions must be iterable.")

    # test that descriptions come from the same dataset
    if not ignore_sampling_configs:
        if any([not descriptions[0].from_same_dataset(desc) for desc in descriptions]):
            raise ValueError(f"Descriptions (and results) must be from similar sampling "\
                "of the dataset. Otherwise, combining the results doesn't make sense.")

    if (not len(ys) == len(y_pred_scoress)) or (not len(ys) == len(descriptions)):
        raise ValueError(f"All iterables passed to the evaluation need to " \
            "have the same length.")

    dataset_description = descriptions[0]
    evaluation_description = EvaluationMultipleSamplesDescription(
        dataset_description=dataset_description,
        algorithm_name=algorithm_name, means=dict(), std_devs=dict(),
        n_steps=len(descriptions))

    for metric in metrics:
        scores = []
        for y, y_pred_scores in zip(ys, y_pred_scoress):
            scores.append(_evaluate_metric(metric, y, y_pred_scores))
        scores = np.array(scores)
        mean, std_dev = np.mean(scores), np.std(scores)
        evaluation_description.means[metric] = mean
        evaluation_description.std_devs[metric] = std_dev

    if print:
        evaluation_description.pretty_print()

    return evaluation_description



def _evaluate_metric(metric: str, y: np.ndarray, y_pred_scores: np.ndarray) -> float:
    """Helper function to evaluate a metric passed as string.

    :param metric: Name of the metric. All available metrics are listed at `all_metrics`
    :param y: Binary ground truth values (0: normals, 1: anomalies)
    :param y_pred_scores: The raw anomaly scores as returned by a fitted model

    :return: Metric score
    """
    if metric not in all_metrics:
        raise ValueError(f"Metric {metric} doesn't exist. Only choose from {all_metrics}.")
    metric_function = metrics_dict[metric]
    return metric_function(y, y_pred_scores)



class EvaluationObject():
    """This object can be used for storing the results of multiple samples
    from the same dataset. The objects needed for evaluation, i.e., the ground
    truth, the predictions and the dataset description, can be stored using
    the `add` method. Once all samples have been added to the object, the
    `evaluate` method can be used to create a :class:`EvaluationMultipleSamplesDescription`
    object which stores data about the dataset and the algorithm's performance
    on the dataset.
    """

    def __init__(self, algorithm_name: Optional[str] = None):
        """Constructor method.
        """
        self.algorithm_name: str = algorithm_name
        self.ground_truths = []
        self.predictions = []
        self.descriptions = []


    def add(self, ground_truth: np.ndarray, prediction: np.ndarray,
        description: AnomalyDatasetDescription) -> None:
        """Stores the result of a sample step.

        :param ground_truth: Binary ground truth values (0: normals, 1: anomalies)
        :param prediction: The raw anomaly scores as returned by a fitted model
        :param description: Dataset description object
        """
        self.ground_truths.append(ground_truth)
        self.predictions.append(prediction)
        self.descriptions.append(description)


    def evaluate(self, print: bool = True,
        metrics: List[str] = ['roc_auc', 'average_precision', 'adjusted_average_precision'],
        ignore_sampling_configs: bool = False) -> EvaluationMultipleSamplesDescription:
        """Evaluate the data of the collected experiments. Note that there
        should be a single evaluation object per dataset, and one object should
        only be used when sampling multiple times with the same parameters from
        the same dataset.

        :param print: Print the result of the evaluation, defaults to True
        :param metrics: List of strings with names of metrics that are to be
            calculated. All available metrics are listed at `all_metrics`, defaults to
            `['roc_auc', 'average_precision', 'adjusted_average_precision']
        :param ignore_sampling_configs: If this is set, the sampling configs
            are not checked for consistency, i.e., it is possible that different
            normal and anomaly labels, different sampling sizes, etc. are used.
            This can be used if multiple different datasets should be considered
            as one, e.g., in the case of MVTec AD. In this case, the first
            sampling config should contain the name of the final dataset, i.e.,
            in this case MVTec AD. Defaults to False
        """
        return evaluate_on_multiple_samples(ys = self.ground_truths,
            y_pred_scoress=self.predictions, descriptions=self.descriptions,
            print=print, metrics=metrics, algorithm_name=self.algorithm_name,
            ignore_sampling_configs=ignore_sampling_configs)


class ComparisonObject():
    """This object allows comparing multiple algorithms on multiple datasets,
    and provides easy-to-use functionality to export findings in different
    formats.

    :param evaluations: List of evaluations that are to be added directly at the
        beginning, defaults to `[]`
    """

    def __init__(self,
        evaluations: List[EvaluationMultipleSamplesDescription] = []):
        """Constructor method"""
        self.evaluations = evaluations
        self.datasets = None # do not restrict datasets that are included into output from the beginning -> None does not exclue anything
        self.algorithms = None # do not restrict algorithms that are included into output from the beginning
        self.metrics = None

    def add_evaluation(self,
        evaluation: EvaluationMultipleSamplesDescription) -> None:
        """Appends an evaluation to the list of evaluations stored in the object.

        :param evaluation: Evaluation to be added
        """
        self.evaluations.append(evaluation)

    def choose_datasets(self, datasets_to_include: List[str] = None,
        verbose: bool = True) -> None:
        """Restrict which datasets should be included into the output. Can also
        be used to determine the order of datasets, which is otherwise defined
        by the order of inserting evaluations.

        :param datasets_to_include: List of datasets that should be included
            into the results visualization. The names need to be the same
            as specified as name in the :class:ClassificationDataset. Defaults
            to all.
        :param verbose: Print which of the specified datasets actually existed
            among the specified ones, defaults to True
        """
        # if all datasets should be included, no need for testing
        if datasets_to_include == None:
            self.datasets = None
        else: # otherwise, specific datasets are specified
            datasets = []
            for dataset in datasets_to_include:
                # check that the dataset is there
                for eval in self.evaluations:
                    # if the dataset exists in the evaluation objects: add dataset,
                    # and stop iterating through evaluation objects
                    if eval.dataset_description.name == dataset:
                        datasets.append(dataset)
                        break
            self.datasets = datasets
            if verbose:
                print(f"The following datasets existed in the data: {self.datasets}")
        return


    def choose_algorithms(self, algorithms_to_include: List[str] = None,
        verbose: bool = True) -> None:
        """Restrict which algorithms should be included into the output. Can
        also be used to determine the order of algorithms, which is otherwise
        defined by the order of inserting evaluations.

        :param algorithms_to_include: List of algorithms that should be included
            into the results visualization. The names need to be the same
            as specified as name in the :class:ClassificationDataset. Defaults
            to all.
        :param verbose: Print which of the specified algorithms actually existed
            among the specified ones, defaults to True
        """
        # if all algorithms should be included, no need for testing
        if algorithms_to_include == None:
            self.algorithms = None
        else: # otherwise, specific algorithms are specified
            algorithms = []
            for algorithm in algorithms_to_include:
                # check that the algorithm is there
                for eval in self.evaluations:
                    # if the algorithm exists in the evaluation objects: add algorithm,
                    # and stop iterating through evaluation objects
                    if eval.algorithm_name == algorithm:
                        algorithms.append(algorithm)
                        break
            self.algorithms = algorithms
            if verbose:
                print(f"The following algorithms existed in the data: {self.algorithms}")
        return

    def choose_metric(self, metrics_to_include: List[str] = None,
        verbose: bool = True) -> None:
        """Restrict which metrics should be included into the output. Can
        also be used to determine the order of metrics, which is otherwise
        defined by the order of inserting evaluations.

        :param metrics_to_include: List of metrics that should be included
            into the results visualization. The names need to be the same
            as specified as name in the :class:ClassificationDataset. Defaults
            to all.
        :param verbose: Print which of the specified metrics actually existed
            among the specified ones, defaults to True
        """
        # if all metrics should be included, no need for testing
        if metrics_to_include == None:
            self.metrics = None
        else: # otherwise, specific metrics are specified
            metrics = []
            for metric in metrics_to_include:
                # check that the metric is there
                for eval in self.evaluations:
                    # if the metric exists in the evaluation objects: add metric,
                    # and stop iterating through evaluation objects
                    if metric in eval.means:
                        metrics.append(metric)
                        break
            self.metrics = metrics
            if verbose:
                print(f"The following metrics existed in the data: {self.metrics}")
        return


    def friedman_nemenyi(self, metric='roc_auc', verbose: bool = True):
        """Calculates, prints and return the Friedman and Nemenyi test.
        Note that the ordering of the algorithms can be seen from the first
        printed table if verbose=True is set.

        :param metric: Which metric to use the results from, defaults to 'roc_auc'
        :param verbose: Print results when calculating, defaults to True

        :return: A tuple with (friedman result, nemenyi result)
        """
        pd_dataframes_dict = self._get_results(split_on_metrics=True)
        df = pd_dataframes_dict[metric][0]
        if verbose:
            print(f"Using the following dataframe:")
            print(df)
        values = df.T.values[:-1, :-1] # removes averages
        friedmanresults = friedmanchisquare(*values)
        if verbose:
            print(f"Friedman results: {friedmanresults}")
        nemenyiresults = posthoc_nemenyi_friedman(values)
        if verbose:
            print("Nemenyi results: (Note that the ordering of the algorithms is the same as in the first table)")
            print(nemenyiresults)
        return friedmanresults, nemenyiresults


    def print_results(self, split_on_metrics: bool = True,
        include_stdevs: bool = False) -> None:
        """Prints the results table, i.e., the result for the specified metrics,
        datasets, and algorithms (all if not specified otherwise).

        :param split_on_metrics: If multiple datasets and multiple metrics are
            chosen, one table per metric is computed. If it is set to `False`,
            split is done on datasets. Defaults to True.
        :param include_stdevs: If set to `True`, the standard deviations are
            also printed. Defaults to `False`.
        """
        pd_dataframes_dict = self._get_results(split_on_metrics=split_on_metrics)

        if not include_stdevs:
            if split_on_metrics:
                for metric in self._get_metrics():
                    print(f"For {metric}:")
                    print(pd_dataframes_dict[metric][0])
            else:
                for dataset in self._get_datasets():
                    print(f"For {dataset}:")
                    print(pd_dataframes_dict[dataset][0])

        else: # do include standard deviations
            if split_on_metrics:
                for metric in self._get_metrics():
                    print(f"For {metric}:")
                    print(self._combine_mean_stdev(*pd_dataframes_dict[metric]))
                    print()
            else:
                raise NotImplementedError(f"print_res0")
        return


    def print_latex(self, split_on_metrics: bool = True,
        include_stdevs: bool = False) -> None:
        """Prints the results table in latex format, i.e., the result for the
        specified metrics, datasets, and algorithms (all if not specified
        otherwise). What is being printed can be directly copied into latex.

        :param split_on_metrics: If multiple datasets and multiple metrics are
            chosen, one latex table per metric is computed. If it is set to `False`,
            split is done on datasets. Defaults to `True`.
        :param include_stdevs: If set to `True`, the standard deviations are
            also printed. Defaults to `False`.
        """
        pd_dataframes_dict = self._get_results(split_on_metrics=split_on_metrics)
        latex_dict = self._convert_results_into_latex(pd_dataframes_dict, include_stdevs)

        if split_on_metrics:
            for metric in self._get_metrics():
                print(f"For {metric}:")
                print(latex_dict[metric])
                print()
        else:
            for dataset in self._get_datasets():
                print(f"For {dataset}:")
                print(pd_dataframes_dict[dataset])
                print()
        return


    def return_latex(self, split_on_metrics: bool = True,
        include_stdevs: bool = False) -> Dict[str, str]:
        """Returns the results table in latex format, i.e., the result for the
        specified metrics, datasets, and algorithms (all if not specified
        otherwise).

        :param split_on_metrics: If multiple datasets and multiple metrics are
            chosen, one latex table per metric is computed. If it is set to `False`,
            split is done on datasets.
        :param include_stdevs: If set to `True`, the standard deviations are
            also printed. Defaults to `False`.

        :return: A dictionary with metrics/datasets as key and their results in
            latex format as value
        """
        pd_dataframes_dict = self._get_results(split_on_metrics=split_on_metrics)
        latex_dict = self._convert_results_into_latex(pd_dataframes_dict, include_stdevs)
        return latex_dict


    def write_latex(self, split_on_metrics: bool = True,
        filepath: str = "evaluation.tex", include_stdevs: bool = False) -> None:
        """Writes the results table in latex format, i.e., the result for the
        specified metrics, datasets, and algorithms (all if not specified
        otherwise), into the specified file

        :param split_on_metrics: If multiple datasets and multiple metrics are
            chosen, one latex table per metric is computed. If it is set to `False`,
            split is done on datasets. Defaults to True.
        :param filepath: Path to the file in which the result tables are stored.
        :param include_stdevs: If set to `True`, the standard deviations are
            also printed. Defaults to `False`.
        """
        pd_dataframes_dict = self._get_results(split_on_metrics=split_on_metrics)
        latex_dict = self._convert_results_into_latex(pd_dataframes_dict, include_stdevs)
        with open(filepath, "w+") as f:
            for key in latex_dict.keys():
                f.write(fr"% latex table for {key}:")
                f.write('\n')
                f.write(latex_dict[key])
                f.write('\n')
                f.write('\n')
        return


    def print_sampling_configs(self):
        """Prints the configurations used for sampling from the datasets when
        comparing multiple algorithms on datasets. What is printed depends on
        the setting, i.e., unsupervised or semisupervised.
        It is also checked that everytime a dataset is used, the same sampling
        configuration is used so that the results really are comparable.
        """
        # 0 PREPARATION
        datasets = self._get_datasets() # list with all datasets that should be considered
        datasets_dict = dict() # dictionary to be converted into pd.DataFrame. Key is dataset name, value is pd.Series describing the dataset

        # tuples for which attributes are collected per dataset and what their attribute name is
        # - note that the attribute name is an iterable
        unsupervised_items = [
            ('n', ['dataset_description', 'number_instances']),
            ('contamination_rate', ['dataset_description', 'contamination_rate']),
            ('sampling_steps', ['n_steps']),
        ]
        semisupervised_items = [
            ('training_n', ['dataset_description', 'number_instances_training']),
            ('training_contamination_rate', ['dataset_description', 'training_contamination_rate']),
            ('test_n', ['dataset_description', 'number_instances_test']),
            ('test_contamination_rate', ['dataset_description', 'test_contamination_rate']),
            ('sampling_steps', ['n_steps']),
        ]

        # 1 COLLECT VALUES into one Series per Dataset -> DataFrame
        for evaluation in self.evaluations:
            if not datasets: # if list with datasets to gather information for is empty, not need to collect further information
                break
            if evaluation.dataset_description.name in datasets: # if we still need information for this dataset, collect it
                # delete dataset from list of datasets as it is no longer needed
                ds_name = evaluation.dataset_description.name
                datasets.remove(ds_name)
                dataset_series_dict = dict() # dict to be converted in pd.Series. Key is feature of dataset, value that features value
                # get list of values to collect (depends on un- or semisupervised)
                if isinstance(evaluation.dataset_description, SemisupervisedAnomalyDatasetDescription): # Semisupervised
                    items_list = semisupervised_items
                else: # unsupervised
                    items_list = unsupervised_items
                # use items list to fill dict
                for key, attrs in items_list:
                    value = evaluation
                    for attr in attrs:
                        value = getattr(value, attr)
                    dataset_series_dict[key] = value
                # make pd.Series and add to datasets dict
                dataset_series = pd.Series(dataset_series_dict)
                datasets_dict[ds_name] = dataset_series

        # 2 CHECK VALUES from all evaluations to be the same
        for evaluation in self.evaluations:
            ds_name = evaluation.dataset_description.name
            dataset_series = datasets_dict[ds_name]
            # get list of items to compare:
            if isinstance(evaluation.dataset_description, SemisupervisedAnomalyDatasetDescription): # Semisupervised
                items_list = semisupervised_items
            else: # unsupervised
                items_list = unsupervised_items
            # compare all items from that list
            for key, attrs in items_list:
                series_value = dataset_series[key]
                evaluation_value = evaluation
                for attr in attrs:
                    evaluation_value = getattr(evaluation_value, attr)
                if not math.isclose(series_value, evaluation_value):
                    raise Exception(f"Evaluation objects are not matching, i.e., not all "\
                        f"results from the same dataset are sampled in the same way. "\
                        f"More specifically, for dataset {ds_name}, expected value for "\
                        f"{key} was {series_value} but got {evaluation_value} on "\
                        f"algorithm {evaluation.algorithm_name}.")

        # 3 MAKE DATAFRAME
        print(pd.DataFrame(datasets_dict))



    def _get_results(self, split_on_metrics: bool = True) -> Dict[str, Tuple[pd.DataFrame]]: #dict if multiple metrics and split_on_metrics or multiple datasets and !split_on_metrics
        """Helper function that transforms the list of evaluations into a
        dictionary where each metric/dataset has two pd.DataFrames associated with
        it: One for the mean, and one for the standard deviation.
        In the mean pd.DataFrame, the results for the specified algorithms and
        datasets/metrics as well as the averages are present.
        In the std_dev pd.DataFrame, standard deviations are stored.

        :param split_on_metrics: If multiple datasets and multiple metrics are
            chosen, one latex table per metric is computed. If it is set to `False`,
            split is done on datasets. Defaults to True.
        """
        included_metrics = self._get_metrics()
        included_datasets = self._get_datasets()
        included_algorithms = self._get_algorithms()
        return_dict = dict()

        if split_on_metrics:
            for metric in included_metrics:
                mean_series_dict = dict() # key is dataset name, value is pd.Series with metrics per algorithm
                stdev_series_dict = dict()
                for dataset in included_datasets:
                    mean_dataset_results_dict = dict() # key is algorithm, value is score
                    stdev_dataset_results_dict = dict()
                    for algorithm in included_algorithms:
                        mean_metric_value = np.nan
                        stdev_metric_value = np.nan
                        for eval in self.evaluations:
                            if (eval.dataset_description.name == dataset) and (eval.algorithm_name == algorithm):
                                mean_metric_value = eval.means.get(metric, np.nan)
                                stdev_metric_value = eval.std_devs.get(metric, np.nan)
                        mean_dataset_results_dict[algorithm] = mean_metric_value
                        stdev_dataset_results_dict[algorithm] = stdev_metric_value
                    mean_series_dict[dataset] = pd.Series(mean_dataset_results_dict)
                    stdev_series_dict[dataset] = pd.Series(stdev_dataset_results_dict)

                mean_dataframe = pd.DataFrame(mean_series_dict)
                stdev_dataframe = pd.DataFrame(stdev_series_dict)
                # add average column (averages over rows)
                mean_dataframe['Average'] = mean_dataframe.apply(lambda row: np.nanmean(row), axis=1)
                # add average row (averages over columns)
                mean_dataframe.loc['Average'] = mean_dataframe.apply(lambda row: np.nanmean(row))
                # remove average of averages
                mean_dataframe.loc['Average', 'Average'] = np.nan
                return_dict[metric] = (mean_dataframe, stdev_dataframe)

        else:
            raise NotImplementedError()

        return return_dict



    def _get_metrics(self):
        """Helper function that returns all metrics used in the evaluation.
        If None are specified, all available metrics are used. Otherwise, only
        those that are specified are included.
        """
        if not (self.metrics == None):
            return self.metrics
        else:
            all_metrics = []
            for eval in self.evaluations:
                for metric in eval.means.keys():
                    if metric not in all_metrics:
                        all_metrics.append(metric)
            return all_metrics


    def _get_datasets(self):
        """Helper function that returns all datasets used in the evaluation.
        If None are specified, all available datasets are used. Otherwise, only
        those that are specified are included.
        """
        if not (self.datasets == None):
            return self.datasets
        else:
            all_datasets = []
            for eval in self.evaluations:
                dataset_name = eval.dataset_description.name
                if dataset_name not in all_datasets:
                    all_datasets.append(dataset_name)
            return all_datasets


    def _get_algorithms(self):
        """Helper function that returns all algorithms used in the evaluation.
        If None are specified, all available algorithms are used. Otherwise, only
        those that are specified are included.
        """
        if not (self.algorithms == None):
            return self.algorithms
        else:
            all_algorithms = []
            for eval in self.evaluations:
                algorithm = eval.algorithm_name
                if algorithm not in all_algorithms:
                    all_algorithms.append(algorithm)
            return all_algorithms


    def _convert_results_into_latex(self, results: Dict[str, pd.DataFrame],
        include_stdevs: bool = False) -> Dict[str, str]:
        """Helper function that converts a dictionary of dataframes into a
        dictionary of latex tables. In each column, the largest element is bold
        and the second largest element is in italics. The keys are the same
        in the input and ouput.
        """
        df_results = results.copy() # copied because there will be in_place changes
        keys = results.keys()
        result_dict = dict()
        for key in keys:
            df = df_results[key][0]
            df_stdevs = df_results[key][1]

            # for each dataframe: Iterate through cols. Remember index of highest
            # and second highest metric per col. Transform col into string, then
            # apply markup
            for column in df.columns:
                col_values_copy = df[column].copy()
                del col_values_copy['Average']
                # get index for highest and second highest element
                first_index, second_index, *_ = col_values_copy.sort_values(ascending=False).index
                # convert into string columns
                df[column] = df[column].map('{:.3f}'.format).astype(str)
                if not (column == 'Average'):
                    df_stdevs[column] = df_stdevs[column].map('{:.3f}'.format).astype(str)


                # is specified, add standard_devs
                if include_stdevs and not (column == 'Average'):
                    for idx in col_values_copy.index:
                        df.loc[idx, column] = df.loc[idx, column] + r'$\pm$' + df_stdevs.loc[idx, column]

                # add markup to highest and second highest element
                # NOTE: By putting this above `if include_stdevs`..., only the mean is highlighted
                df.loc[first_index, column] = fr"\textbf{{{df.loc[first_index, column]}}}"
                df.loc[second_index, column] = fr"\textit{{{df.loc[second_index, column]}}}"

            # in the second step, the dataframes with the correct markup as values
            # are transformed into a string representation of the table
            def _make_tabular_ending_correct(string: str) -> str:
                string = string[:-2]
                return string + r"\\"

            strs = []
            strs.append(r"\begin{center}")
            c_format = " c"*len(df.columns) + " c "
            strs.append(fr"\begin{{tabular}}{{ {c_format} }}")

            # add heading with algorithm names
            string_builder = "  & "
            for col in df.columns:
                string_builder = string_builder + col + " & "
            string_builder = _make_tabular_ending_correct(string_builder)
            strs.append(string_builder)

            # add actual data
            for row_index in df.index:
                # string builder gradually builds up the line
                string_builder = "  "
                row = df.loc[row_index]
                # first append name of the row
                string_builder = string_builder + row_index + " & "
                # and the row elements
                for row_element in row:
                    if row_element == 'nan':
                        string_builder = string_builder + "   & "
                    else:
                        string_builder = string_builder + row_element + " & "
                string_builder = _make_tabular_ending_correct(string_builder)
                strs.append(string_builder)
            strs.append(r"\end{tabular}")
            strs.append(r"\end{center}")

            res = '\n'.join(strs)
            res = res.replace('_', r'\_')
            result_dict[key] = res
        return result_dict


    def _combine_mean_stdev(self, mean_df: pd.DataFrame, stdev_df: pd.DataFrame) -> pd.DataFrame:
        """Helper that takes a pd.DataFrame filled with mean values and one
        with standard deviation values and combines them so that "mean +- stdev"
        is in the cells of the returned pd.DataFrame.

        :param mean_df: pd.DataFrame with mean values
        :param stdev_df: pd.DataFrame with standard deviations

        :return: A pd.DataFrame with "mean +- stdev" for all except for average
            cells
        """
        means = mean_df.copy()
        stdevs = stdev_df.copy()

        for col in stdevs.columns:

            means[col] = means[col].map('{:.3f}'.format).astype(str)
            stdevs[col] = stdevs[col].map('{:.3f}'.format).astype(str)

            for idx in stdevs.index:
                means.loc[idx, col] = means.loc[idx, col] + r'+-' + stdevs.loc[idx, col]
        return means
