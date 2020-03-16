import numpy as np


def dRMSE(y_true, y_pred, freq):
    """
        Compute the root-mean-squared error of the predicted variations.
        :param y_true: ground truth of shape (n_days, None)
        :param y_pred: predictions of shape (n_days, None)
        :param freq: sampling frequency in minutes (e.g., 5)
        :return: dRMSE
    """
    dy_true = np.diff(y_true) / freq
    dy_pred = np.diff(y_pred) / freq
    return np.sqrt(np.mean((dy_true - dy_pred) ** 2))
