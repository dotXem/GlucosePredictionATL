from misc.utils import locate_model, locate_params
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


class T_SNE(object):
    def __init__(self, source_dataset, target_dataset, target_subject, exp_name, model_name, params):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.target_subject = target_subject
        self.exp_name = exp_name

        self.model_name = model_name
        self.Model = locate_model(model_name)
        self.params = locate_params(params)

        print(self.Model, self.params)

        self.ph = misc.constants.ph_f
        self.hist = self.params["hist"] // misc.constants.freq
        self.day_len = misc.constants.day_len_f

        self.train, self.valid, self.test, _ = preprocessing_source_multi(self.source_dataset, self.target_dataset,
                                                                          self.target_subject, self.ph, self.hist,
                                                                          self.day_len)

    def plot(self, pca_components, tsne_components, perplexity=30, learning_rate=200.0, split_source=0,
             split_target=None):
        tsne_features, y = self.compute_tsne_features(pca_components, tsne_components, perplexity, learning_rate,
                                                      split_source, split_target)

        plt.figure()
        for i in range(int(max(y[:, 1]) + 1)):
            plt.plot(*tsne_features[y[:, 1] == i].transpose(), "*")

        plt.show()

    def save(self, pca_components, tsne_components, split_source=0, split_target=None):
        features, y = self.compute_tsne_features(pca_components, tsne_components, split_source, split_target)

        dir = os.path.join(misc.datasets.path, "results", self.source_dataset + "_2_" + self.target_dataset, "TSNE",
                           self.exp_name)
        if not os.path.exists(dir): os.makedirs(dir)

        for i in range(int(max(y[:, 1]) + 1)):
            feat = features[y[:, 1] == i]
            file_name = "tsne_" + self.target_dataset + self.target_subject + "_" + str(i)
            np.savetxt(os.path.join(dir, file_name), feat)

    def compute_min_average_distance(self, dist2other, pca_components, tsne_components, perplexity=30,
                                     learning_rate=200.0, split_source=0,
                                     split_target=None, use_tsne=True):

        if use_tsne:
            feat, y = self.compute_tsne_features(pca_components, tsne_components, perplexity, learning_rate,
                                                 split_source, split_target)
        else:
            feat, y = self.compute_features(split_source, split_target)

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
                               learning_rate=200.0, split_source=0,
                               split_target=None):
        if use_tsne:
            feat, y = self.compute_tsne_features(pca_components, tsne_components, perplexity, learning_rate,
                                                 split_source, split_target)
        else:
            feat, y = self.compute_features(split_source, split_target)
        # printd("feat done")
        dist = np.zeros((len(feat), len(feat)))
        other_class = np.zeros((len(feat), len(feat)), dtype=bool)
        # printd("init done")
        for i in range(len(feat)):
            dist[i + 1:, i] = np.sum((feat[i + 1:] - feat[i]) ** 2, axis=1)
            other_class[i + 1:, i] = y[i + 1:, 1] != y[i, 1]
        # printd("loop done")
        dist = dist + dist.T
        other_class = other_class + other_class.T
        dist[dist == 0] = np.inf
        min_all = np.min(dist, axis=0)
        dist_other = dist.copy()
        dist_other[~other_class] = np.inf
        min_other_class = np.min(dist_other, axis=0)

        return np.mean(min_other_class / min_all)

    def compute_features(self, split_source=0, split_target=None):
        file_path = self._compute_file_path()
        model = self.Model(self.target_subject, self.ph, self.params, self.train[0], self.valid[0], self.test[0])
        features, _ = model.extract_features("valid", file_path)
        y = self.valid[0].loc[:, ["y", "domain"]].values

        # source_data = source_preprocessing(self.source_dataset, self.target_dataset, self.target_subject)
        # _, valid, _, _, _ = source_data.get_split(split_source)
        # if split_target is not None:
        #     target_data = target_preprocessing(self.target_dataset, self.target_subject)
        #     _, target_valid, _, _, _ = target_data.get_split(split_target)
        #     target_index = max(valid[1][:, 1]) + 1
        #     valid = [np.r_[valid[0], target_valid[0]],
        #              np.r_[valid[1], np.c_[target_valid[1], np.full((len(target_valid[1]), 1), target_index)]]]
        #
        # file_path = self._compute_file_path(split_source)
        # model = self.Model(self.params)
        # model.load_weights_from_file(file_path)
        #
        # features, y = model.extract_features(*valid)

        return features, y

    def compute_tsne_features(self, pca_components, tsne_components, perplexity=30, learning_rate=200.0, split_source=0,
                              split_target=None):

        features, y = self.compute_features(split_source, split_target)

        pca_features = PCA(pca_components, random_state=misc.constants.seed).fit_transform(features)
        tsne_features = TSNE(tsne_components, perplexity=perplexity, learning_rate=learning_rate,
                             random_state=misc.constants.seed).fit_transform(pca_features)

        return tsne_features, y

    def compute_perplexity(self, n_neighbours, pca_components, tsne_components, tsne_perplexity=30, learning_rate=200.0,
                           split_source=0, split_target=None, use_tsne=True):

        if use_tsne:
            feat, y = self.compute_tsne_features(pca_components, tsne_components, tsne_perplexity, learning_rate,
                                                 split_source, split_target)
        else:
            feat, y = self.compute_features(split_source, split_target)

        dist = np.zeros((len(feat), len(feat)))
        for i in range(len(feat)):
            dist[i + 1:, i] = np.sum((feat[i + 1:] - feat[i]) ** 2, axis=1)
        dist = dist + dist.T

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

    def compute_perplexity2(self, n_neighbours, pca_components, tsne_components, tsne_perplexity=30,
                            learning_rate=200.0,
                            split_source=0, split_target=None, use_tsne=True):

        """get the tsne features"""

        if use_tsne:
            feat, y = self.compute_tsne_features(pca_components, tsne_components, tsne_perplexity, learning_rate,
                                                 split_source, split_target)
        else:
            feat, y = self.compute_features(split_source, split_target)

        """ compute the distance in n dimensional space between every points """

        # duplicate feat along first axis (n_samples, tsne_components) to (n_samples, n_samples, tsne_components)
        # print(feat.shape)
        # a = np.ones((1, 1, len(feat)))
        # c = np.expand_dims(feat.transpose(1, 0), 2)
        # print(a.shape, c.shape)
        # prod = (c.astype(np.float16) @ a.astype(np.float16)).transpose(2, 1, 0)
        # print(prod.shape)
        #
        # # extract element of comparison and reformat
        # x1 = np.expand_dims(prod.diagonal().transpose(1, 0), 1)
        # # REMOVED - delete element of comparison from list of points
        # # x2 = prod[~np.eye(prod.shape[0], dtype=bool)].reshape(prod.shape[0], -1, prod.shape[2])
        # x2 = prod
        # dist = np.sum((x2 - x1) ** 2, axis=2)
        #
        # """ get the n nearest neighbours """
        # min_indexes = dist.argsort(axis=1)[:, 1:n_neighbours + 1]  # first index is always it self
        # mask = np.zeros((prod.shape[:2]), dtype=bool)
        # for i in range(len(min_indexes)):
        #     for j in range(len(min_indexes[i])):
        #         mask[i][min_indexes[i, j]] = True
        # feat_neighbours = prod[mask].reshape(prod.shape[0],-1,prod.shape[2])

        subfeat_sample_len = 100
        # perplexities = []
        # for j in range(len(feat) // subfeat_sample_len):
        #     subfeat = feat[j*subfeat_sample_len:(j+1)*subfeat_sample_len]
        #     suby = y[j*subfeat_sample_len:(j+1)*subfeat_sample_len]
        #     a = np.ones((1, 1, len(subfeat)))
        #     c = np.expand_dims(feat.transpose(1, 0), 2)
        #     prod = (c.astype(np.float16) @ a.astype(np.float16)).transpose(2, 1, 0)
        #     x1 = np.expand_dims(prod.diagonal().transpose(1, 0), 1)
        #     x2 = prod
        #     dist = np.sum((x2 - x1) ** 2, axis=2)
        #     min_indexes = dist.argsort(axis=1)[:, 1:n_neighbours + 1]
        #
        #     for i in range(len(subfeat)):
        #         # dist = np.sum((feat - feat[i]) ** 2, axis=1)
        #         # min_indexes = dist.argsort()[1:n_neighbours + 1]
        #
        #
        #         """ compute the count for every class """
        #         classes = y[min_indexes[i], 1]
        #         # classes = y[min_indexes, 1]
        #         count_dict = {a: 0 for a in np.arange(np.max(y[:, 1]) + 1).astype(str)}
        #         for class_ in classes:
        #             count_dict[str(class_)] += 1
        #
        #         """ compute the probability for every class """
        #         total = sum(count_dict.values())
        #         probabilities = np.array(list(count_dict.values())) / total
        #         probabilities = probabilities[np.nonzero(probabilities)]
        #         """ compute the sample perplexity """
        #
        #         perplexity_sample = 2 ** (- np.sum(probabilities * np.log2(probabilities)))
        #         perplexities.append(perplexity_sample)

        """ for every sample... """
        perplexities = []
        for i in range(len(feat)):
            dist = np.sum((feat - feat[i]) ** 2, axis=1)
            min_indexes = dist.argsort()[1:n_neighbours + 1]

            """ compute the count for every class """
            # classes = y[min_indexes[i], 1]
            classes = y[min_indexes, 1]
            count_dict = {a: 0 for a in np.arange(np.max(y[:, 1]) + 1).astype(str)}
            for class_ in classes:
                count_dict[str(class_)] += 1

            """ compute the probability for every class """
            total = sum(count_dict.values())
            probabilities = np.array(list(count_dict.values())) / total
            probabilities = probabilities[np.nonzero(probabilities)]
            """ compute the sample perplexity """

            perplexity_sample = 2 ** (- np.sum(probabilities * np.log2(probabilities)))
            perplexities.append(perplexity_sample)

        """ compute the mean perplexity """
        perplexity = np.mean(perplexities)

        """ normalize mean perplexity ? """
        perplexity_norm = perplexity / (np.max(y[:, 1]) + 1)

        return perplexity, perplexity_norm

    def _compute_file_path(self):
        file_name = self.model_name + "_" + self.target_dataset + self.target_subject + ".pt"
        return os.path.join("processing", "models", "weights", self.source_dataset + "_2_" + self.target_dataset,
                            self.exp_name, file_name)
