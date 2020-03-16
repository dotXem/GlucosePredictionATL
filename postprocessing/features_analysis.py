import _misc
from postprocessing.tsne import T_SNE
import numpy as np
from tools.printd import printd

class FeaturesAnalyzer():

    tsne_components = 2
    tsne_pca_components = 50
    tsne_perplexity = 30.0
    tsne_lr = 200.0

    def __init__(self, source_ds, target_ds, exp_name, model_name):
        self.source_ds = source_ds
        self.target_ds = target_ds
        self.exp_name = exp_name
        self.model_name = model_name
        self.target_subjects = _misc.datasets_subjects_dict[self.target_ds]

    def perplexity(self, n_neighbours, use_tsne=True, save_file=None):
        perplexity = []
        for target_subject in self.target_subjects:
            # if target_subject == "5":
            printd("Perplexity " + self.target_ds + target_subject)
            tsne = T_SNE(self.source_ds, self.target_ds, target_subject, self.exp_name, self.model_name)
            for split_source in range(_misc.cv):
                perplexity.append(tsne.compute_perplexity(n_neighbours,
                                                          self.tsne_pca_components,
                                                          self.tsne_components,
                                                          self.tsne_perplexity,
                                                          self.tsne_lr,
                                                          split_source=split_source,
                                                          use_tsne=use_tsne))
            #printd(np.mean(perplexity,axis=0))
        self.save(save_file, perplexity)
        
        return np.mean(perplexity,axis=0)

    def distance(self, to_other=False, use_tsne=True, save_file=None):
        dpr = []
        for target_subject in self.target_subjects:
            # if target_subject == "5":
            printd("Distance " + self.target_ds + target_subject)
            tsne = T_SNE(self.source_ds, self.target_ds, target_subject, self.exp_name, self.model_name)
            for split_source in range(_misc.cv):
                dpr.append(tsne.domain_proximity_ratio(use_tsne,
                                                          self.tsne_pca_components,
                                                          self.tsne_components,
                                                          self.tsne_perplexity,
                                                          self.tsne_lr,
                                                          split_source=split_source))

        self.save(save_file, dpr)

        return np.mean(dpr), np.std(dpr)


    def save(self, save_file, data):
        if save_file is not None:
            np.save(save_file, data)

