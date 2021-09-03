import numpy as np
import tensorflow as tf
from typing import Dict

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lof import LOF


class AELOF():
    """
    Unsupervised anomaly detection model using an AutoEncoder (AE) for dimensionality
    reduction and feeding the reduced-dimensionality data points into LOF.
    The AE and LOF are imported from PyOD (https://pyod.readthedocs.io/).

    Note:
    - Parameters for the AE and LOF are the same as for PyOD'd models.
      They are passed to the constructor as a dictionary.
    - If the AE's hidden_neurons are specified, the iteratble needs to contain
      4 integers. Otherwise, the model needs to be changed in that the
      bottleneck representation is the output of another layer.

    EXAMPLE:
    (x, y), config = anomaly_detection_dataset.sample(...)

    ae_parameters = {'hidden_neurons': [6, 3, 3, 6], 'hidden_activation': 'LeakyReLU', 'verbose': 0, 'random_state': 42}
    lof_parameters = {'n_neighbors': 50}
    ae_lof = AELOF(AE_parameters=ae_parameters, LOF_parameters=lof_parameters)
    ae_lof.fit(x)
    preds = ae_lof.decision_scores_
    """

    def __init__(self, AE_parameters: Dict, LOF_parameters: Dict,
        random_state: int = 42):
        """
        Constructor for AE+LOF anomaly detection algorithm.

        :param AE_parameters: Dictionary with parameters for PyOD's AutoEncoder
        :param LoF_parameters: Dictionary with parameters for PyOD's LOF
        :param random_state: Random seed to ensure reproducibility
        """
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        self.ae = AutoEncoder(**AE_parameters)
        self.lof = LOF(**LOF_parameters)

    def fit(self, X: np.ndarray):
        """
        Fits the anomaly detection algorithm to the data and calculates
        self.decision_scores_.

        :param X: Data points
        """
        # first fit autoencoder
        self.ae.fit(X)

        # retrieve bottleneck representation
        features_bottleneck_model = tf.keras.models.Model(
            inputs=self.ae.model_.inputs,
            outputs=self.ae.model_.get_layer(name=self.ae.model_.layers[8].name).output
        )
        bottleneck_repr = features_bottleneck_model(X, training=False).numpy()

        # pass bottleneck representation to LOF
        self.lof.fit(bottleneck_repr)

        # set decision_scores_ to LOF's decision_scores_
        self.decision_scores_ = self.lof.decision_scores_
