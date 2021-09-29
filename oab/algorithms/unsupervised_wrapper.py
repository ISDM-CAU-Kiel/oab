import numpy as np
import yaml
from typing import Dict

from oab.algorithms.abstract_classes import AbstractWrapperFromRecipe, AbstractWrapperToRecipe

class UnsupervisedWrapperFromRecipe(AbstractWrapperFromRecipe):

    def __init__(self):
        pass

    def run_algo(self, x: np.ndarray, obj, yaml_path):
        """
        """
        # load yaml
        with open(yaml_path, "r") as stream:
            yaml_content = yaml.safe_load(stream)

        # load method and attribute names and parameters
        init_params = yaml_content['init']['params']
        fit_method = yaml_content['fit']['method_name']
        fit_params = yaml_content['fit']['params']
        decision_scores_field = yaml_content['decision_scores']['field_name']

        # initalize
        algo = obj(**init_params)

        # fit
        getattr(algo, fit_method)(x, **fit_params)

        # return decision functions
        return getattr(algo, decision_scores_field)


class UnsupervisedWrapperToRecipe(AbstractWrapperToRecipe):

    def __init__(self):
        pass

    def track_init(self, obj, params: Dict = {}):
        self.init_dict = {'params': params}
        return obj(**params)

    def track_fit(self, x, obj, params: Dict = {}, fit_method: str = 'fit'):
        self.fit_dict = {
            'method_name': fit_method,
            'params': params
        }
        return getattr(obj, fit_method)(x, **params)

    def track_decision_scores(self, obj, field_name: str = 'decision_scores_'):
        self.decision_scores_dict = {'field_name': field_name}
        return getattr(obj, field_name)

    def store_recipe(self, yaml_path):
        yaml_content = {
            'init': self.init_dict,
            'fit': self.fit_dict,
            'decision_scores': self.decision_scores_dict
        }
        with open(yaml_path, "w+") as stream:
             yaml.dump(yaml_content, stream)
