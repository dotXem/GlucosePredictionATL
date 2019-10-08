import numpy as np


def RMSE(y_true, y_pred):
    """
        Compute the root-mean-squared error of the predictions
        :param y_true: ground truth of shape (n_days, None)
        :param y_pred: predictions of shape (n_days, None)
        :return: root-mean-squared error
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
