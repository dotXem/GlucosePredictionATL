from scipy.stats import sem, t
import numpy as np
from .results import ResultsSubject
from misc.datasets import datasets


class ResultsSignificance():
    def __init__(self, model, model_ref, exp, exp_ref, scenario_group):
        self.model = model
        self.model_ref = model_ref
        self.exp = exp
        self.exp_ref = exp_ref
        self.scenario_group = scenario_group

        self.res = self.load_results(self.exp, self.scenario_group, self.model)
        self.res_ref = self.load_results(self.exp_ref, self.scenario_group, self.model_ref)

    def load_results(self, exp, scenario_group, model):
        results_l = []
        for source_dataset, target_dataset in zip(*self.scenario_group_2_datasets(scenario_group)):
            for subject in datasets[target_dataset]["subjects"]:
                exp_ds = source_dataset + "_2_" + target_dataset + "\\" + exp
                results_l.append(ResultsSubject(model, exp_ds, 30,target_dataset, subject).compute_results(raw_score=True))

        return results_l

    def scenario_group_2_datasets(self, scenario_group):
        split = scenario_group.split("_")
        scenario = split[0]
        add_T = split[1] if len(split) == 2 else None

        if scenario == "intra":
            source_datasets, target_datasets = ["idiab", "ohio"], ["idiab", "ohio"]
        elif scenario == "inter":
            source_datasets, target_datasets = ["idiab", "ohio"], ["ohio", "idiab"]
        elif scenario == "synth":
            source_datasets, target_datasets = ["t1dms", "t1dms"], ["idiab", "ohio"]
        elif scenario == "augm":
            source_datasets, target_datasets = ["idiab+ohio", "idiab+ohio"], ["idiab", "ohio"]

        if add_T == "T" and not scenario == "synth":
            source_datasets = [source_dataset + "+t1dms" for source_dataset in source_datasets]

        return source_datasets, target_datasets

    def compute_significance(self, metric, confidence):
        scores = self.compute_paired_ratio(metric).ravel()

        n = len(scores)
        m = np.mean(scores)
        std_err = sem(scores)
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)

        ci_start = m - h
        ci_end = m + h

        return m, h, ci_start, ci_end

    def compute_paired_ratio(self, metric):
        res = np.reshape([res_sbj[metric] for res_sbj in self.res], (-1,1))
        res_ref = np.reshape([res_sbj[metric] for res_sbj in self.res_ref], (-1,1))
        return res / res_ref
