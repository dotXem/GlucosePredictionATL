from _misc import day_len_freq, ph_freq, hist_freq
import numpy as np

def _x_y_split(data, index):
    return [[split[:, :index] for split in data], [split[:, index:] for split in data]]


def _reshape_data_into_days(data):
    n_days = len(data.index) // day_len_freq
    days = [
        data.iloc[d * day_len_freq:(d + 1) * day_len_freq, [1,2,3]].reset_index(
            drop=True)
        for d in range(n_days)
    ]
    return days


def _create_samples(data):
    samples = []
    for day in data:
        # create the samples by adding the past values accounting for the amount of history
        X = np.array([
            day.iloc[j: j + day_len_freq - hist_freq].values for j in
            range(hist_freq)
        ])
        y_pred = X[-1, ph_freq:, 0].reshape(-1, 1)

        # the for loop is done the other way around to speed up, but we need to transpose after
        X = np.rollaxis(X[:, :-ph_freq, :], 1, 0)
        X = np.reshape(X, (X.shape[0], -1))

        samples.append(np.concatenate([X, y_pred], axis=1))

    return samples


def _merge_subjects(data):
    return [np.concatenate(split, axis=0) for split in data]


def _merge_days(data):
    return [np.concatenate(split, axis=0) for split in data]


def _merge_pool_subjects(data):
    return np.array([[np.concatenate(split) for split in set] for set in data])


def _per_subject_merge_days(data):
    return np.array([[np.concatenate(subject, axis=0) for subject in split] for split in data])


def _dann_add_domains(data):
    def add_domain(subject_data, domain):
        domain_array = np.reshape([domain] * len(subject_data), (-1, 1))
        return np.c_[subject_data, domain_array]

    data = [[add_domain(subject, domain) for subject, domain in zip(split, np.arange(len(split)))] for split in data]

    return data

