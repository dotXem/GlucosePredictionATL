import pandas as pd
import numpy as np
import _misc


def _standardize(train, other, mean_l, std_l):
    standardized_datasets = []
    for dataset in ([train] + other):
        standardized_splits = []
        for split, mean, std in zip(dataset, mean_l, std_l):
            standardized_splits.append((split - mean) / std)
        standardized_datasets.append(standardized_splits)

    return standardized_datasets


def _compute_means_stds(train_sets):
    mean_l, std_l = [], []
    for train_split in train_sets:
        mean_l.append(np.mean(train_split, axis=0))
        std_l.append(np.std(train_split, axis=0))
    return mean_l, std_l


def _standardize_pool(train, other):
    mean_l, std_l = _compute_means_stds(train)
    standardized_datasets = _standardize(train, other, mean_l, std_l)
    return standardized_datasets, mean_l, std_l


def _per_subject_standardization(train, valid):
    train, valid = [np.rollaxis(set,1,0) for set in [train, valid]]

    data = [_standardize_pool(train_sbj, [valid_sbj])
            for train_sbj, valid_sbj in zip(train, valid)]

    # data = [_standardize_pool(train_sbj, [valid_sbj])[0]
    #         for train_sbj, valid_sbj in zip(train, valid)]
    # data = np.transpose(data, (1, 2, 0))  # set first

    data, mean, std = np.rollaxis(np.array(data), 1, 0)
    data = np.transpose(np.array(list(data)), (1, 2, 0))  # order = set, subject, split
    mean = np.transpose(np.array(list(mean)), (1, 0, 2))  # order = split, subject, X
    std = np.transpose(np.array(list(std)), (1, 0, 2))  # order = split, subject, X

    return data, mean, std


def standardize_features(train, other):
    def reshape_features(data):
        return np.c_[data]

    train = reshape_features(train)
    other = [reshape_features(other_) for other_ in other]

    mean, std = np.mean(train, axis=0), np.std(train, axis=0)
    train = (train - mean) / std
    other = [(other_ - mean) / std for other_ in other]

    def reshape_features_back(data):
        return [data[:, :-1], data[:, -1:]]

    std_sets = [reshape_features_back(set) for set in [train, *other]]

    return std_sets


def _T1DMS_scaling(data):
    # scale insulin from pmol to unit
    data.loc[:, "insulin"] = data.loc[:, "insulin"] / 6000.0

    # accumulate the CHO intakes
    CHO_indexes = data[np.invert(data.loc[:, "CHO"] == 0.0)].index
    meals, meal, start_idx, past_idx = [], data.loc[CHO_indexes[0],"CHO"], CHO_indexes[0], CHO_indexes[0]
    for idx in CHO_indexes[1:]:
        if idx == past_idx+1:
            meal = meal + data.loc[idx, "CHO"]
        else:
            meals.append([start_idx, meal])
            meal = data.loc[idx, "CHO"]
            start_idx = idx
        past_idx = idx
    meals.append([start_idx, meal])
    meals = np.array(meals)

    data.loc[:, "CHO"] = 0.0
    data.loc[meals[:,0],"CHO"] = meals[:,1]

    # resample 5 minutes
    data["datetime"] = pd.to_datetime(data["datetime"])
    resampler = data.resample("5min", on="datetime")

    data_resampled = pd.DataFrame()
    data_resampled["datetime"] = resampler["glucose"].mean().index
    data_resampled["glucose"] = resampler["glucose"].mean().values
    data_resampled["insulin"] = resampler["insulin"].sum().values
    data_resampled["CHO"] = resampler["CHO"].sum().values

    # discard first day of simulation
    data_resampled = data_resampled.loc[_misc.day_len_freq:]

    return data_resampled
