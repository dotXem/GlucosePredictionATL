import os
import misc.datasets
import misc.constants
import matplotlib.pyplot as plt
import numpy as np
from misc.utils import printd
from preprocessing.preprocessing import preprocessing_source_multi
from misc.utils import locate_model, locate_params
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from .metrics.local_domain_perplexity import local_domain_perplexity


class FeaturesAnalyzer():
    def __init__(self, source_ds, target_ds, weights_name, model_name, params_name):
        """
        Object that is used to analyze the features outputed by saved models pre-transfer
        (models trained on the multi sources, before finetuning). Can be use to plot and save t-SNE representations,
        and compute the local domain perplexity metric.
        :param source_ds: source dataset (e.g., "idiab", "ohio", "idiab+t1dms", etc.)
        :param target_ds: target dataset (i.e., "idiab" or "ohio")
        :param weights_name: name of the weights
        :param model_name: name of the model
        :param params_name: name of the used parameters
        """
        self.source_dataset = source_ds
        self.target_dataset = target_ds
        self.exp_name = weights_name
        self.model_name = model_name
        self.Model = locate_model(model_name)
        self.params = locate_params(params_name)
        self.target_subjects = misc.datasets.datasets[self.target_dataset]["subjects"]

        self.ph = misc.constants.ph_f
        self.hist = self.params["hist"] // misc.constants.freq
        self.day_len = misc.constants.day_len_f

    def _compute_features(self, target_subject, split):
        """
        Compute the features for a dataset excluding the given target subject, for a given split
        :param target_subject: name or number of target subject to exclude during computation
        :param split: number of the split
        :return: features, domains (class of subject)
        """
        train, valid, test, _ = preprocessing_source_multi(self.source_dataset, self.target_dataset, target_subject,
                                                           self.ph, self.hist, self.day_len)
        file_path = self._compute_file_path(target_subject)
        model = self.Model(target_subject, self.ph, self.params, train[split], valid[split], test[split])
        features, _ = model.extract_features("valid", file_path)
        y = valid[split].loc[:, ["y", "domain"]].values[:, 1]

        return features, y

    def _compute_tsne_features(self, features, tsne_components=2, pca_components=50, perplexity=30.0,
                               learning_rate=200.0):
        """
        Compute the t-SNE features from the raw features by performing a PCA to reduce the initial space to a lower
        space (e.g., 50), and by then performing the t-SNE reduction to an even lower one (e.g., 2).
        :param features: raw features of shape (n_samples, n_features)
        :param tsne_components: dimension of the t-SNE reduction
        :param pca_components: dimension of the first reduction by PCA
        :param perplexity: perplexity
        :param learning_rate: learning rate
        :return: tsne features in the shape (n_samples, tsne_components)
        """
        pca_features = PCA(pca_components, random_state=misc.constants.seed).fit_transform(features)
        tsne_features = TSNE(tsne_components, perplexity=perplexity, learning_rate=learning_rate,
                             random_state=misc.constants.seed).fit_transform(pca_features)

        return tsne_features

    def plot_tsne(self, target_subject):
        """
        Plot the t-SNE representation of a given target subject (split 0)
        :param target_subject: name/number of target subject (thus excluded from the plot)
        :return: /
        """
        features, domains = self._compute_features(target_subject, split=0)
        features = self._compute_tsne_features(features)

        plt.figure()
        for i in range(int(max(domains) + 1)):
            plt.plot(*features[domains == i].transpose(), "*")

        plt.show()

    def local_domain_perplexity(self, n_neighbours, reduce_tsne=False, save_file=None):
        """
        Compute the local domain perplexity metric (LDP) for every target subjects and splits
        :param n_neighbours: size of neighbourhood
        :param reduce_tsne: if the features need to be reduced to 2D with t-SNE
        :param save_file: if the end results should be saved
        :return: mean and std of the LDP metrics
        """
        ldp_arr = []
        for target_subject in self.target_subjects:
            printd("Perplexity " + self.target_dataset + target_subject)

            for split in range(misc.constants.cv):
                features, domains = self._compute_features(target_subject, split)
                if reduce_tsne:
                    features = self._compute_tsne_features(features)

                ldp_arr.append(local_domain_perplexity(features, domains, n_neighbours))

        if save_file is not None:
            np.save(save_file, ldp_arr)

        return np.mean(ldp_arr, axis=0), np.std(ldp_arr, axis=0)

    def _compute_file_path(self, target_subject):
        """
        Convenient function to compute the path to the weights of the models
        :param target_subject: name/number of the target subject (the one who is set aside)
        :return: file path
        """
        file_name = self.model_name.upper() + "_" + self.target_dataset + target_subject + ".pt"
        return os.path.join(misc.constants.path, "processing", "models", "weights",
                            self.source_dataset + "_2_" + self.target_dataset,
                            self.exp_name, file_name)
