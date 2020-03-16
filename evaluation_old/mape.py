import numpy as np

def MAPE(y_true, y_pred):
    """
        Compute the fitness of the predictions, with is a normalized RMSE, expressed in %
        :param y_true: ground truth of shape (n_days, None)
        :param y_pred: predictions of shape (n_days, None)
        :return: fitness
    """
    return 100 * np.mean(np.abs((y_true-y_pred)/y_true))