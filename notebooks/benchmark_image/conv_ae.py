"""Using Convolutional Auto Encoder with Outlier Detection
"""
# Author of Auto Encoder: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause
# https://pyod.readthedocs.io/en/latest/_modules/pyod/models/auto_encoder.html#AutoEncoder
# lines marked with #* are from this file
# code blocks in #*<start> ... #*<end> as well

import numpy as np
import tensorflow as tf
#*<start>
from sklearn.preprocessing import StandardScaler

# if tensorflow 2, import from tf directly
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import mean_squared_error
from tensorflow import keras
#*<end>


class ConvAutoEncoder():
    def __init__(self, latent_dim: int,
                 hidden_activation='relu', output_activation='sigmoid',
                 loss=mean_squared_error, optimizer='adam',
                 epochs=100, batch_size=32,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=42, contamination=0.1):
        self.latent_dim = latent_dim
        #*<start>
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        #*<end>
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

    def _build_model(self):
        """
        NEEDS
        - self.input_shape
        - self.latent_dim
        - self.strides, self.kernel_size
        - self.target_shape_decoder
        - self.units_in_decoder

        """

        model = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=self.input_shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=self.kernel_size, strides=self.strides,
                activation='relu', padding='same'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=self.kernel_size, strides=self.strides,
                activation='relu', padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_dim),
            # tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
            tf.keras.layers.Dense(units=self.units_in_decoder, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(self.target_shape_decoder)),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=self.kernel_size,
                strides=self.strides, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=self.kernel_size,
                strides=self.strides, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=self.input_shape[2],
                kernel_size=self.kernel_size,
                strides=1, padding='same', activation='relu'),
            ]
        )

        #*<start>
        # Compile model
        model.compile(loss=self.loss, optimizer=self.optimizer)
        if self.verbose >= 1:
            print(model.summary())
        return model
        #*<end>



    def compute_target_shape_decoder(self, input_shape, strides):
        """HELPER"""
        dim1 = input_shape[0] // (strides*strides)
        return (dim1, dim1, 64)



    def fit(self, X, y=None):
        """
        """
        # print(X.shape)
        if len(X.shape) == 3:
            # print(f"Old shape {X.shape}")
            X = X.reshape(list(X.shape) + [1])
            # print(f"New shape {X.shape}")



        self.n_samples_, self.input_shape = X.shape[0], X.shape[1:]
        if self.input_shape[0] > 100:
            self.kernel_size, self.strides = 5, 4
        else:
            self.kernel_size, self.strides = 3, 2
        self.target_shape_decoder = self.compute_target_shape_decoder(self.input_shape, self.strides)
        self.units_in_decoder = self.target_shape_decoder[0] * self.target_shape_decoder[1] * self.target_shape_decoder[2]


        # Standardize data for better performance
        if self.preprocessing: #*
            self.scaler_ = StandardScaler() #*
            original_dims = X.shape
            X_norm = self.scaler_.fit_transform(X.reshape(original_dims[0], -1)).reshape(original_dims)
        else: #*
            X_norm = np.copy(X) #*

        #*<start>
        # Shuffle the data for validation as Keras do not shuffling for
        # Validation Split
        np.random.shuffle(X_norm)

        # Build AE model & fit with X
        self.model_ = self._build_model()

        self.history_ = self.model_.fit(X_norm, X_norm,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose).history
        # Reverse the operation for consistency
        # Predict on X itself and calculate the reconstruction error as
        # the outlier scores. Noted X_norm was shuffled has to recreate
        if self.preprocessing:
        #*<end>
            original_dims = X.shape
            X_norm = self.scaler_.fit_transform(X.reshape(original_dims[0], -1)).reshape(original_dims)
        else: #*
            X_norm = np.copy(X) #*

        pred_scores = self.model_.predict(X_norm) #*
        self.decision_scores_ = pairwise_distances_no_broadcast(
            X_norm.reshape((X_norm.shape[0], -1)),
            pred_scores.reshape((X_norm.shape[0], -1)))
        return self #*

    #*<start>
    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        #*<end>
        # print(X.shape)
        if len(X.shape) == 3:
            # print(f"Old shape {X.shape}")
            X = X.reshape(list(X.shape) + [1])
            # print(f"New shape {X.shape}")

        if self.preprocessing: #*
            original_dims = X.shape
            X_norm = self.scaler_.fit_transform(X.reshape(original_dims[0], -1)).reshape(original_dims)
        #*<start>
        else:
            X_norm = np.copy(X)

        # Predict on X and return the reconstruction errors
        pred_scores = self.model_.predict(X_norm)
        return pairwise_distances_no_broadcast(
            X_norm.reshape((X_norm.shape[0], -1)),
            pred_scores.reshape((X_norm.shape[0], -1))
        )

##############

def pairwise_distances_no_broadcast(X, Y):
    """Utility function to calculate row-wise euclidean distance of two matrix.
    Different from pair-wise calculation, this function would not broadcast.

    For instance, X and Y are both (4,3) matrices, the function would return
    a distance vector with shape (4,), instead of (4,4).

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples

    Y : array of shape (n_samples, n_features)
        Second input samples

    Returns
    -------
    distance : array of shape (n_samples,)
        Row-wise euclidean distance of X and Y
    """
    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError("pairwise_distances_no_broadcast function receive"
                         "matrix with different shapes {0} and {1}".format(
            X.shape, Y.shape))
    return _pairwise_distances_no_broadcast_helper(X, Y)


def _pairwise_distances_no_broadcast_helper(X, Y):  # pragma: no cover
    """Internal function for calculating the distance with numba. Do not use.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples

    Y : array of shape (n_samples, n_features)
        Second input samples

    Returns
    -------
    distance : array of shape (n_samples,)
        Intermediate results. Do not use.

    """
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()
