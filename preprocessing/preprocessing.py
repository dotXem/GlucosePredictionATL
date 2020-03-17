import pandas as pd
from preprocessing.preprocessing_t1dms import preprocessing_t1dms
from preprocessing.preprocessing_ohio import preprocessing_ohio
import misc.datasets
from misc.utils import printd

preprocessing_per_dataset = {
    "t1dms_adult": preprocessing_t1dms,
    "t1dms_adolescent": preprocessing_t1dms,
    "t1dms_child": preprocessing_t1dms,
    "ohio": preprocessing_ohio,
}


def preprocessing(dataset, subject, ph, hist, day_len, tl_mode):
    """
    associate every dataset with a specific pipeline - which should be consistent with the others

    :param dataset: name of dataset (e.g., "ohio")
    :param subject: name of subject (e.g., "559")
    :param ph: prediction horizon in minutes (e.g., 5)
    :param hist: length of history in minutes (e.g., 60)
    :param day_len: typical length of a day in minutes standardized to the sampling frequency (e.g. 288 for 1440 min at freq=5 minutes)
    :return: train, valid, test folds
    """
    if "target" in tl_mode:
        printd("Preprocessing " + dataset + subject + "...")
        return preprocessing_per_dataset[dataset](dataset, subject, ph, hist, day_len)
    elif "source" in tl_mode:
        return preprocessing_source(dataset, ph, hist, day_len)

def preprocessing_source(dataset, ph, hist, day_len):
        train_ds, valid_ds, test_ds, scalers_ds = [], [], [], []
        for i, subject in enumerate(misc.datasets.datasets[dataset]["subjects"]):
            printd("Preprocessing " + dataset + subject + "...")
            train_sbj, valid_sbj, test_sbj, scalers_sbj = preprocessing_per_dataset[dataset](dataset, subject, ph, hist,
                                                                                             day_len)

            train = pd.concat([train_sbj[0], valid_sbj[0]]).sort_values("datetime")
            valid = test_sbj[0]
            test = test_sbj[0]

            train["domain"] = i
            valid["domain"] = i
            test["domain"] = i

            train_ds.append(train)
            valid_ds.append(valid)
            test_ds.append(test)
            scalers_ds.append(scalers_sbj[0])

        train_ds, valid_ds, test_ds = [pd.concat(ds) for ds in [train_ds, valid_ds, test_ds]]

        return [train_ds], [valid_ds], [test_ds], scalers_ds
