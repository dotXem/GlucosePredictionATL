from scipy.signal import correlate
import numpy as np

def time_lag(y_true, y_pred, freq):
    """
        Compute the time-lag (TL) metric, as the shifting number maximizing the correlation
        between the prediction and the ground truth.
        :param y_true: ground truth of shape (n_days, None)
        :param y_pred: predictions of shape (n_days, None)
        :return: mean of daily time-lags
    """

    lags = []
    for y_true_d, y_pred_d in zip(y_true, y_pred):
        y_true_d_norm = (y_true_d - np.mean(y_true_d)) / np.std(y_true_d)
        y_pred_d_norm = (y_pred_d - np.mean(y_pred_d)) / np.std(y_pred_d)
        lags.append(len(y_true_d_norm) - np.argmax(correlate(y_true_d_norm, y_pred_d_norm)) - 1)
    return freq * np.mean(lags)


def diabetes_time_lag(y_true, y_pred, freq):
    def get_event_start_indexes(day, event_type):
        if event_type == "hypo":
            indexes = np.where(day <= 70)[0]
        elif event_type == "hyper":
            indexes = np.where(day >= 180)[0]

        events = []
        event_tmp = []
        for i in range(len(indexes) - 1):
            if indexes[i + 1] == indexes[i] + 1:
                event_tmp.append(indexes[i])
            else:
                if len(event_tmp) >= 3:
                    # remember event
                    events.append(event_tmp)
                event_tmp = []

        if len(event_tmp) >= 3:
            events.append(event_tmp)

        return np.min(events, axis=1)

    for y_true_d, y_pred_d in zip(y_true, y_pred):
        hypo_true, hyper_true = get_event_start_indexes(y_true_d, "hypo"), get_event_start_indexes(y_true_d, "hyper")
        hypo_pred, hyper_pred = get_event_start_indexes(y_pred_d, "hypo"), get_event_start_indexes(y_pred_d, "hyper")

    #TODO work on or remove

    pass