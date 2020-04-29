import numpy as np


def domain_relative_distance(features, y):
    """
    Domains Relative Distance (DRD): it measures the proximity of samples from different domains. It is computed as
    the average ratio between the minimal distance between samples of different domains and the minimal distance b
    etween any sample. If the ratio is small, it means that samples from different domains are in average close to
    each other in the features space. On the other hand, if the ratio is big, those samples are far from each other in
    the features space.
    :param features: samples' features of shape (n_samples, n_featuers)
    :param y: samples' domains of shape (n_samples)
    :return: DRD metric
    """
    dist = np.zeros((len(features), len(features)))
    other_class = np.zeros((len(features), len(features)), dtype=bool)
    for i in range(len(features)):
        dist[i + 1:, i] = np.sum((features[i + 1:] - features[i]) ** 2, axis=1)
        other_class[i + 1:, i] = y[i + 1:, 1] != y[i, 1]
    dist = dist + dist.T
    other_class = other_class + other_class.T
    dist[dist == 0] = np.inf

    min_all = np.min(dist, axis=0)
    dist_other = dist.copy()
    dist_other[~other_class] = np.inf
    min_other_class = np.min(dist_other, axis=0)
    res = np.mean(min_other_class / min_all)
    return res
