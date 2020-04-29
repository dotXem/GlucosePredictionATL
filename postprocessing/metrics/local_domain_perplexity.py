import numpy as np
from scipy.spatial import distance

def local_domain_perplexity(features, domains, N):
    """
    Local Domain Perplexity (LDP): it measures in average, from 0 to 1, how uniform the distribution of the features'
    domains in their close neighbourhood. Whereas a high DNP implies that the features are general accross all the
    domains, a low DNP means that features are specific to the domain.
    :param features: samples' features of shape (n_samples, n_featuers)
    :param domains: samples' domains of shape (n_samples)
    :param N: size of the neighbourhood
    :return: LDP metric
    """
    dist = distance.cdist(features, features, 'euclidean')

    min_indexes = dist.argsort()[:, 1:N + 1]
    classes = domains[min_indexes]

    classes_count = np.zeros((len(classes), int(np.max(domains)) + 1))
    for i in range(len(classes)):
        for neighbour_class in classes[i]:
            classes_count[i, int(neighbour_class)] += 1

    probabilities = classes_count / N
    probabilities[probabilities == 0] = 1e-25
    probabilities = probabilities / np.sum(probabilities, axis=1).reshape(-1, 1)

    perplexities = 2 ** (- np.sum(probabilities * np.log2(probabilities), axis=1))

    nnp = np.mean(perplexities) / (np.max(domains) + 1)

    return nnp