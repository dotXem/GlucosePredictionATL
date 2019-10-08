import misc
import numpy as np


def compute_subjects_list(dataset, subject):
    """
        From the name of a dataset and a subject, compute the list of [dataset, subject].
        :param dataset: name of the dataset, accept "all"
        :param subject: name of the subject, accept "all"
        :return: the list of [dataset, subject]
    """
    datasets_subjects_dict = misc.datasets_subjects_dict
    if dataset is not None:
        if dataset == "all":
            datasets_subjects = np.array(
                np.concatenate([[[k, v_] for v_ in v] for k, v in datasets_subjects_dict.items()], axis=0)).reshape(-1,
                                                                                                                    2)
        else:
            if subject is not None:
                if subject == "all":
                    datasets_subjects = np.array(
                        [[dataset, v_] for v_ in datasets_subjects_dict[dataset]]).reshape(-1, 2)
                else:
                    datasets_subjects = np.array([[dataset, subject]])
            else:
                datasets_subjects = np.array([[dataset, datasets_subjects_dict[dataset]]])
    else:
        datasets_subjects = np.array(
            [[datasets_subjects_dict.keys()[0], datasets_subjects_dict[datasets_subjects_dict.keys()[0]]]])

    return datasets_subjects
