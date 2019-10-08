import numpy as np

def fit(y_true, y_pred):
    """
        Compute the fitness of the predictions, with is a normalized RMSE, expressed in %
        :param y_true: ground truth of shape (n_days, None)
        :param y_pred: predictions of shape (n_days, None)
        :return: fitness
    """
    return (1 - (np.sqrt(np.mean((y_true-y_pred)**2)))/(np.sqrt(np.mean((y_true-np.mean(y_true))**2)))) * 100