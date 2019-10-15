import os
from misc import path

idiab_subjects = ["1", "2", "3", "4", "5"]
ohio_subjects = ["559", "563", "570", "575", "588", "591"]


def _compute_dataset_files(dataset_name, subject_names):
    return [_compute_subject_file(dataset_name, subject_name) for subject_name in subject_names]


def _compute_subject_file(dataset_name, subjet_name):
    return os.path.join(path, "data", "dynavolt", dataset_name, dataset_name + "_subject" + subjet_name + ".csv")


idiab_files = _compute_dataset_files("IDIAB", idiab_subjects)
ohio_files = _compute_dataset_files("Ohio", ohio_subjects)

subject_files = idiab_files + ohio_files
# subject_files = idiab_files


def _compute_subject_full_name(dataset_name, subject_name):
    return dataset_name + "_subject" + subject_name


def _remove_subject_from_pool(source_dataset, target_dataset, target_subject):
    subject_files = _select_pool_files(source_dataset)
    subject_full_name = _compute_subject_full_name(target_dataset, target_subject)
    for file in subject_files:
        if subject_full_name in file:
            subject_files.remove(file)
            return subject_files
    return subject_files

def _select_pool_files(source_dataset):
    if source_dataset == "IDIAB":
        return idiab_files
    elif source_dataset == "Ohio":
        return ohio_files
    elif source_dataset == "all":
        return idiab_files + ohio_files


def _get_subject_file_from_name(dataset_name, subject_name):
    subject_full_name = _compute_subject_full_name(dataset_name, subject_name)
    for file in subject_files:
        if subject_full_name in file:
            return file
