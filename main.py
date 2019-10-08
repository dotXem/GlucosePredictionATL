from pydoc import locate
import sys
import argparse
import os
from misc import path, cv, freq
from preprocessing.preprocessing import source_preprocessing, target_preprocessing
from postprocessing.postprocessing import source_postprocessing, target_postprocessing
from evaluation.results import ResultsSource, ResultsTarget
from tools.printd import printd


def main_source(source_dataset, target_dataset, target_subject, Model, params, eval, split, save_file):
    data = source_preprocessing(source_dataset, target_dataset, target_subject)

    splits = [split] if split is not None else range(cv)
    results = []
    for split_number in splits:
        train, valid, test, means, stds = data.get_split(split_number)

        model = Model(params)
        model.fit(*train, *valid)

        y_true, y_pred = model.predict(*test) if eval == "test" else model.predict(*valid)

        results.append(source_postprocessing(y_true, y_pred, means, stds))

        if save_file is not None:
            model.save_encoder_predictor(os.path.join(save_file, "_", str(split_number)))

    results = ResultsSource(save_file, source_dataset, target_dataset, target_subject, freq, results=results)
    printd(results.get_results())

    if save_file is not None:
        results.save()


def main_target(target_dataset, target_subject, Model, params, mode, pretraining, split, eval, save, plot):
    pass


if __name__ == "__main__":
    """
        --mode: 4 modes 
                "source_training":      train a model on source dataset minus the target subject
                "target_training":      train a model on the target subject only
                "target_global":        use a model trained with the "source_training" mode to make the prediction for the 
                                        target subject. --pretraining_file must be set.
                "target_finetuning":    finetune a model trained with the "source_training" mode on the target subject
        --source_dataset:
                dataset used in the "source_training" mode, can be either "IDIAB", "Ohio" or "all"
        --target_dataset and --target_subject:
                specify the subject used in the "target_X" modes and removed from the "source_training" if needed
        --model:
                specify the model used in all the modes
        --pretraining:
                specify the files to be used in the "target_global" and "target_finetuning" modes
        --eval:
                specify the evaluation set to be used, in the "target_X" modes, either "valid" or "test". default:
                "valid".
        --split:
                restrict the training/evaluation to one specific split in [0-11] for "traget_X" modes and [0-3] in 
                "source_training" mode. default: None
        --log:
                specify the file where the logs shall be redirected to. default: None
        --save:
                if set, saves the prediction results in the specified folder ("target_X" modes) and the model checkpoints
                in the "source_training_mode". default: False
        --plot:
                if set, plot the results after the training. default: True  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--source_dataset", type=str)
    parser.add_argument("--target_dataset", type=str)
    parser.add_argument("--target_subject", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--pretraining", type=str)
    parser.add_argument("--eval", type=str)
    parser.add_argument("--split", type=int)
    parser.add_argument("--log", type=str)
    parser.add_argument("--save_file", type=str)
    parser.add_argument("--plot", type=bool)
    args = parser.parse_args()

    model_name = args.model if args.model is not None else sys.exit(-1)
    Model = locate("models." + model_name + "." + model_name)
    params = locate("models." + model_name + ".params")

    # redirect the logs to a file if specified
    if args.log is not None:
        log_file = args.log
        log_path = os.path.join(path, "logs", log_file)
        sys.stdout = open(log_path, "w")

    if args.mode == "source_training":
        main_source(args.source_dataset, args.target_dataset, args.target_subject, Model, params, args.eval, args.split,
                    args.save_file)
    elif args.mode in ["target_training", "target_global", "target_finetuning"]:
        pass

    #
    # def compute_domain2domain_name(pool, dataset):
    #     return pool + "_2_" + dataset
    #
    #
    # d2d_name = compute_domain2domain_name(args.pool, args.dataset)
    #
    #
    # save = args.save if args.save is not None else False
    #
    # model_name = args.model if args.model is not None else sys.exit(-1)
    # Model = locate("models.features_extractors." + model_name + "." + model_name)
    # params = locate("models.features_extractors." + model_name + ".params")
    #
    # files_dir = args.files_dir if args.files_dir is not None else "test"
    # files_dir = os.path.join(d2d_name, files_dir)
    #
    # main_fine_tuning(dataset_name=args.dataset,
    #                  subject_name=args.subject,
    #                  Model=Model,
    #                  params=params,
    #                  files_dir=files_dir,
    #                  split_number=args.split,
    #                  save=save)
