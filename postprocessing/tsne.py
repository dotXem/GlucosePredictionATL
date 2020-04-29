from scipy.spatial import distance
from misc.utils import locate_model, locate_params
from misc.utils import printd
import misc.constants
import misc.datasets
import os
# from preprocessing_old.preprocessing import target_preprocessing, source_preprocessing
from preprocessing.preprocessing import preprocessing_source_multi
import numpy as np
from pydoc import locate
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from timeit import default_timer as timer

#TODO remove
class T_SNE(object):
    def __init__(self, source_dataset, target_dataset, target_subject, exp_name, model_name, params):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.target_subject = target_subject
        self.exp_name = exp_name

        self.model_name = model_name
        self.Model = locate_model(model_name)
        self.params = locate_params(params)

        self.ph = misc.constants.ph_f
        self.hist = self.params["hist"] // misc.constants.freq
        self.day_len = misc.constants.day_len_f

        self.train, self.valid, self.test, _ = preprocessing_source_multi(self.source_dataset, self.target_dataset,
                                                                          self.target_subject, self.ph, self.hist,
                                                                          self.day_len)

    def plot(self, pca_components, tsne_components, perplexity=30, learning_rate=200.0):
        tsne_features, y = self.compute_tsne_features(pca_components, tsne_components, perplexity, learning_rate)

        plt.figure()
        for i in range(int(max(y[:, 1]) + 1)):
            plt.plot(*tsne_features[y[:, 1] == i].transpose(), "*")

        plt.show()

    def save(self, pca_components, tsne_components):
        features, y = self.compute_tsne_features(pca_components, tsne_components)

        dir = os.path.join(misc.constants.path, "results", self.source_dataset + "_2_" + self.target_dataset, "TSNE",
                           self.exp_name)
        if not os.path.exists(dir): os.makedirs(dir)

        for i in range(int(max(y[:, 1]) + 1)):
            feat = features[y[:, 1] == i]
            file_name = "tsne_" + self.target_dataset + self.target_subject + "_" + str(i)
            np.savetxt(os.path.join(dir, file_name), feat)

    def compute_min_average_distance(self, dist2other, pca_components, tsne_components, perplexity=30,
                                     learning_rate=200.0, use_tsne=True):

        if use_tsne:
            feat, y = self.compute_tsne_features(pca_components, tsne_components, perplexity, learning_rate)
        else:
            feat, y = self.compute_features()

        min_distances = []
        for i in range(len(feat)):
            # compute distance of point i to every other point not belonging to same class
            mask = np.ones(feat.shape[0], dtype=np.bool)
            mask[i] = False

            point = feat[i]
            cluster = y[i, 1]
            other_points = feat[mask]
            other_clusters = y[mask][:, 1]

            if dist2other:
                other_points = other_points[np.invert(other_clusters == cluster)]

            distance = np.sqrt(np.sum((other_points - point) ** 2, axis=1))
            min_distances.append(np.min(distance))

        return np.mean(min_distances), np.std(min_distances)

    def domain_proximity_ratio(self, use_tsne=True, pca_components=50, tsne_components=2, perplexity=30,
                               learning_rate=200.0):
        if use_tsne:
            feat, y = self.compute_tsne_features(pca_components, tsne_components, perplexity, learning_rate)
        else:
            feat, y = self.compute_features()

        dist = np.zeros((len(feat), len(feat)))
        other_class = np.zeros((len(feat), len(feat)), dtype=bool)
        for i in range(len(feat)):
            dist[i + 1:, i] = np.sum((feat[i + 1:] - feat[i]) ** 2, axis=1)
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

    def compute_features(self):
        file_path = self._compute_file_path()
        model = self.Model(self.target_subject, self.ph, self.params, self.train[0], self.valid[0], self.test[0])
        features, _ = model.extract_features("valid", file_path)
        y = self.valid[0].loc[:, ["y", "domain"]].values

        return features, y

    def compute_tsne_features(self, pca_components, tsne_components, perplexity=30, learning_rate=200.0):

        features, y = self.compute_features()

        pca_features = PCA(pca_components, random_state=misc.constants.seed).fit_transform(features)
        tsne_features = TSNE(tsne_components, perplexity=perplexity, learning_rate=learning_rate,
                             random_state=misc.constants.seed).fit_transform(pca_features)

        return tsne_features, y

    def compute_perplexity(self, n_neighbours, pca_components, tsne_components, tsne_perplexity=30, learning_rate=200.0,
                           use_tsne=True):

        if use_tsne:
            feat, y = self.compute_tsne_features(pca_components, tsne_components, tsne_perplexity, learning_rate)
        else:
            feat, y = self.compute_features()

        dist = distance.cdist(feat, feat, 'euclidean')

        min_indexes = dist.argsort()[:, 1:n_neighbours + 1]
        classes = y[:, 1][min_indexes]

        classes_count = np.zeros((len(classes), int(np.max(y[:, 1])) + 1))
        for i in range(len(classes)):
            for neighbour_class in classes[i]:
                classes_count[i, int(neighbour_class)] += 1

        probabilities = classes_count / n_neighbours
        probabilities[probabilities == 0] = 1e-25
        probabilities = probabilities / np.sum(probabilities, axis=1).reshape(-1, 1)

        perplexities = 2 ** (- np.sum(probabilities * np.log2(probabilities), axis=1))

        nnp = np.mean(perplexities) / (np.max(y[:, 1]) + 1)

        return nnp

    def _compute_file_path(self):
        file_name = self.model_name.upper() + "_" + self.target_dataset + self.target_subject + ".pt"
        return os.path.join(misc.constants.path, "processing", "models", "weights", self.source_dataset + "_2_" + self.target_dataset,
                            self.exp_name, file_name)
