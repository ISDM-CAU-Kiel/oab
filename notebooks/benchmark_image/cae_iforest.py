import numpy as np
import tensorflow as tf
from typing import Dict

from conv_ae import ConvAutoEncoder
from pyod.models.iforest import IForest

class CAEIForest():
    """
    Semisupervised anomaly detection model using a Convolutional AutoEncoder (CAE)
    for dimensionality reduction and feeting the reduced-dimensionality data
    points into IForest. The CAE is trained on the training set, as is IForest.
    IForest is imported from PyOD (https://pyod.readthedocs.io/).
    """


    def __init__(self, CAE_parameters: Dict, IForest_parameters: Dict,
        random_state: int = 42):
        """
        Constructor for CAE+IForest anomaly detection algorithm.

        :param CAE_parameters: Dictionary with parameters for CAE
        :param IForest_parameters: Dictionary with parameters for PyOD's IForest
        :param random_state: Random seed to ensure reproducibility, defaults to 42
        """
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        self.cae = ConvAutoEncoder(**CAE_parameters)
        self.iforest = IForest(**IForest_parameters)

    def fit(self, X: np.ndarray):
        """
        Fits the semisupervised algorithms. First fits the CAE and then fits the
        IForest with the bottleneck representation of data points. The decision
        scores are then taken from the IForest.

        :param X: Data points
        """
        self.cae.fit(X)
        self.features_bottleneck_model = tf.keras.models.Model(
            inputs=self.cae.model_.inputs,
            outputs=self.cae.model_.get_layer(name=self.cae.model_.layers[3].name).output
        )
        X_new = self.get_bottleneck_representation(X)
        self.iforest.fit(X_new)
        self.decision_scores_ = self.iforest.decision_scores_


    def decision_function(self, X: np.ndarray):
        """
        Computes the anomaly scores of the input X. First, it is passed to the
        CAE for dimensionality reduction. The new representation is then passed
        to the IForest.

        :param X: Data points
        """
        X_new = self.get_bottleneck_representation(X)
        return self.iforest.decision_function(X_new)


    def get_bottleneck_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Helper to get bottleneck representation of X.
        """
        return self.features_bottleneck_model(X, training=False).numpy()
