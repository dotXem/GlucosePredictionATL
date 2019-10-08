import numpy as np


def r(y_true, y_pred):
    """
        Compute the correlation coefficient of the predictions
        :param y_true: ground_truth of shape (n_days, None)
        :param y_pred: predictions of shape (n_days, None)
        :return: correlation coefficient
    """
    numerator = np.sum((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred)))
    denominator = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2)) * np.sqrt(np.sum((y_pred - np.mean(y_pred)) ** 2))
    return numerator / denominator
