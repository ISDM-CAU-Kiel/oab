import numpy as np
import yaml
from typing import Dict

from oab.algorithms.abstract_classes import AbstractWrapperFromRecipe, AbstractWrapperToRecipe

class SemisupervisedWrapperFromRecipe(AbstractWrapperFromRecipe):

    def __init__(self):
        pass

    def run_algo(self, X_train: np.ndarray, X_test: np.ndarray, obj, yaml_path):
        """
        """
        # load yaml
        with open(yaml_path, "r") as stream:
            yaml_content = yaml.safe_load(stream)

        # load method and attribute names and parameters
        init_params = yaml_content['init']['params']
        fit_method = yaml_content['fit']['method_name']
        fit_params = yaml_content['fit']['params']
        predict_method = yaml_content['decision_function']['method_name']
        predict_params = yaml_content['decision_function']['params']

        # initalize
        algo = obj(**init_params)

        # fit
        getattr(algo, fit_method)(X_train, **fit_params)

        # return decision functions
        return getattr(algo, predict_method)(X_test, **predict_params)


class SemisupervisedWrapperToRecipe(AbstractWrapperToRecipe):

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

    def track_decision_function(self, x ,obj, params: Dict = {},
        decision_function_method: str = 'decision_function'):
        self.decision_function_dict = {
            'method_name': decision_function_method,
            'params': params
        }
        return getattr(obj, decision_function_method)(x, **params)

    
    def store_recipe(self, yaml_path):
            yaml_content = {
                'init': self.init_dict,
                'fit': self.fit_dict,
                'decision_function': self.decision_function_dict
            }
            with open(yaml_path, "w+") as stream:
                 yaml.dump(yaml_content, stream)
