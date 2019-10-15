import misc
import os
from preprocessing.preprocessing import target_preprocessing, source_preprocessing
import numpy as np
from pydoc import locate
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class T_SNE(object):
    def __init__(self, source_dataset, target_dataset, target_subject, exp_name, model_name):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.target_subject = target_subject
        self.exp_name = exp_name

        self.model_name = model_name
        self.Model = locate("models." + model_name + "." + model_name)
        self.params = locate("models." + model_name + ".params")

    def plot(self, pca_components, tsne_components, split_source=0, split_target=None):
        tsne_features, y = self.compute_tsne_features(pca_components, tsne_components, split_source, split_target)

        plt.figure()
        for i in range(int(max(y[:, 1]) + 1)):
            plt.plot(*tsne_features[y[:, 1] == i].transpose(), "*")

        plt.show()

    def save(self, pca_components, tsne_components, split_source=0, split_target=None):
        features, y = self.compute_tsne_features(pca_components, tsne_components, split_source, split_target)

        dir = os.path.join(misc.path, "results", self.source_dataset + "_2_" + self.target_dataset, "TSNE",
                           self.exp_name)
        if not os.path.exists(dir): os.makedirs(dir)

        for i in range(int(max(y[:, 1]) + 1)):
            feat = features[y[:, 1] == i]
            file_name = "tsne_" + self.target_dataset + self.target_subject + "_" + str(i)
            np.savetxt(os.path.join(dir, file_name), feat)

    def compute_tsne_features(self, pca_components, tsne_components, split_source=0, split_target=None):
        source_data = source_preprocessing(self.source_dataset, self.target_dataset, self.target_subject)
        _, valid, _, _, _ = source_data.get_split(split_source)
        if split_target is not None:
            target_data = target_preprocessing(self.target_dataset, self.target_subject)
            _, target_valid, _, _, _ = target_data.get_split(split_target)
            target_index = max(valid[1][:, 1]) + 1
            valid = [np.r_[valid[0], target_valid[0]],
                     np.r_[valid[1], np.c_[target_valid[1], np.full((len(target_valid[1]), 1), target_index)]]]

        file_path = self._compute_file_path(split_source)
        model = self.Model(self.params)
        model.load_weights_from_file(file_path)

        features, y = model.extract_features(*valid)

        pca_features = PCA(pca_components, random_state=misc.seed).fit_transform(features)
        tsne_features = TSNE(tsne_components, random_state=misc.seed).fit_transform(pca_features)

        return tsne_features, y

    def _compute_file_path(self, split):
        file_name = self.model_name + "_" + self.target_dataset + self.target_subject + "_" + str(split) + ".pt"
        return os.path.join(self.source_dataset + "_2_" + self.target_dataset, self.exp_name, file_name)
