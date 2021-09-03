import numpy as np
import tensorflow as tf
from typing import Dict

from conv_ae import ConvAutoEncoder
from pyod.models.knn import KNN

class CAEKNN():
    """
    Semisupervised anomaly detection model using a Convolutional AutoEncoder (CAE)
    for dimensionality reduction and feeting the reduced-dimensionality data
    points into KNN. The CAE is trained on the training set, as is KNN.
    KNN is imported from PyOD (https://pyod.readthedocs.io/).
    """


    def __init__(self, CAE_parameters: Dict, KNN_parameters: Dict,
        random_state: int = 42):
        """
        Constructor for CAE+KNN anomaly detection algorithm.

        :param CAE_parameters: Dictionary with parameters for CAE
        :param KNN_parameters: Dictionary with parameters for PyOD's KNN
        :param random_state: Random seed to ensure reproducibility, defaults to 42
        """
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        self.cae = ConvAutoEncoder(**CAE_parameters)
        self.knn = KNN(**KNN_parameters)

    def fit(self, X: np.ndarray):
        """
        Fits the semisupervised algorithms. First fits the CAE and then fits the
        KNN with the bottleneck representation of data points. The decision
        scores are then taken from KNN.

        :param X: Data points
        """
        self.cae.fit(X)
        self.features_bottleneck_model = tf.keras.models.Model(
            inputs=self.cae.model_.inputs,
            outputs=self.cae.model_.get_layer(name=self.cae.model_.layers[3].name).output
        )
        X_new = self.get_bottleneck_representation(X)
        self.knn.fit(X_new)
        self.decision_scores_ = self.knn.decision_scores_


    def get_bottleneck_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Helper to get bottleneck representation of X.
        """
        return self.features_bottleneck_model(X, training=False).numpy()
