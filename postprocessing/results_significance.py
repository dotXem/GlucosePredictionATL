from scipy.stats import sem, t
import numpy as np
from .results import ResultsSubject
import misc.datasets


class ResultsSignificance:

    def __init__(self, model, model_ref, exp, exp_ref, scenario):
        """
        Class that computes the significance of the results for two set of model/experiments for a given metric in the
        shape of confidence intervals.
        :param model: name of model to be compared
        :param model_ref: name of reference model, with which the model will be compared to
        :param exp: name of experiment to be compared
        :param exp_ref: name of reference experiment, with which the model will be compared to
        :param scenario: name of scenario, either "intra", "inter", "synth", any combination concatenated by "+", or "all"
        """
        self.model = model
        self.model_ref = model_ref
        self.exp = exp
        self.exp_ref = exp_ref
        self.scenario = scenario
        self.scenario_2_datasets = self._create_scenario_2_datasets_dict()

        self.res = self.load_results(self.exp, self.scenario, self.model)
        self.res_ref = self.load_results(self.exp_ref, self.scenario, self.model_ref)

    def _create_scenario_2_datasets_dict(self):
        """
        Create a convenient dictionary to retrieve the source and target datasets given specific scenarios.
        Scenarios implemented :
            - intra : source and target patients are from the same dataset
            - inter : source and target patients are from different but real dataset
            - synth : source patients are from synthetic virtual dataset
            - any combination of the former 3
            - all : all merged together
            - training : similar to intra, used to retrieve results that do not use transfer learning (baselines)
        :return:
        """
        scenario_2_datasets = {
            "intra": [["idiab", "idiab"], ["ohio", "ohio"]],
            "inter": [["ohio", "idiab"], ["idiab", "ohio"]],
            "synth": [["t1dms", "idiab"], ["t1dms", "ohio"]],
            "intra+inter": [["idiab+ohio", "idiab"], ["idiab+ohio", "ohio"]],
            "intra+synth": [["idiab+t1dms", "idiab"], ["ohio+t1dms", "ohio"]],
            "inter+synth": [["ohio+t1dms", "idiab"], ["idiab+t1dms", "ohio"]],
            "intra+inter+synth": [["idiab+ohio+t1dms", "idiab"], ["idiab+ohio+t1dms", "ohio"]],
        }

        scenario_2_datasets["all"] = np.concatenate([scenario_2_datasets[key] for key in scenario_2_datasets.keys()],
                                                    axis=0)
        scenario_2_datasets["training"] = scenario_2_datasets["intra"]

        return scenario_2_datasets

    def load_results(self, exp, scenario, model):
        """
        Load the results into an array
        :param exp: experiment name
        :param scenario: scenario, either "intra", "inter", "synth", or any combination of them concatened by "+"
        :param model: model name
        :return: array of results
        """
        results_l = []

        if "training" in exp:
            datasets = self.scenario_2_datasets["training"]
        else:
            datasets = self.scenario_2_datasets[scenario]

        for source_dataset, target_dataset in datasets:
            for subject in misc.datasets.datasets[target_dataset]["subjects"]:
                exp_ds = source_dataset + "_2_" + target_dataset + "\\" + exp
                results_l.append(
                    ResultsSubject(model, exp_ds, 30, target_dataset, subject).compute_raw_results())

        return results_l

    def compute_significance(self, metric, confidence):
        """
        Compute the significance (confidence interval) of the improvement of one set of results over another one for a
        given metric and confidence.
        :param metric: metric used for the significance test
        :param confidence: confidence in the change of performances
        :return: mean confidence, delta, start of interval, end of interval
        """
        scores = self.compute_paired_ratio(metric).ravel()

        n = len(scores)
        m = np.mean(scores)
        std_err = sem(scores)
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)

        ci_start = m - h
        ci_end = m + h

        return m, h, ci_start, ci_end

    def compute_paired_ratio(self, metric):
        """
        Compute the paired ratio between scores in the given metric.
        :param metric: metric that select the scores (see ResultsSubject class)
        :return: paired ratio
        """
        res = np.reshape([res_sbj[metric] for res_sbj in self.res], (-1, 1))
        res_ref = np.reshape([res_sbj[metric] for res_sbj in self.res_ref], (-1, 1))
        return res / res_ref
