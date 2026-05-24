import numpy as np

class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_pred - y_true)**2)

    @staticmethod
    def gradient(y_true, y_pred):
        return 2 * (y_pred - y_true) / (y_true.shape[0] * y_true.shape[1])

class CrossEntropy:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        # Add a small epsilon to avoid log(0)
        eps = 1e-9
        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

    @staticmethod
    def gradient(y_true, y_pred):
        eps = 1e-9
        return -y_true / (y_pred + eps) / y_true.shape[0]
