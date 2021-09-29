import numpy as np
from abc import ABC, abstractmethod

class AbstractWrapperFromRecipe(ABC):

    @abstractmethod
    def run_algo(self, x: np.ndarray, obj, yaml_path):
        pass


class AbstractWrapperToRecipe(ABC):

    @abstractmethod
    def track_init(self, obj, params):
        pass

    @abstractmethod
    def track_fit(self, x, obj, params, fit_method):
        pass

    @abstractmethod
    def store_recipe(self, yaml_path):
        pass
