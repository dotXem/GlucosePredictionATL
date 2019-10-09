from pydoc import locate
import sys
import argparse
import os
from misc import path, cv, freq
from preprocessing.preprocessing import source_preprocessing, target_preprocessing
from postprocessing.postprocessing import source_postprocessing, target_postprocessing
from evaluation.results import ResultsSource, ResultsTarget
from tools.printd import printd
from tools.domain2domain import domain_2_domain_str


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
            file = os.path.join(domain_2_domain_str(source_dataset, target_dataset), save_file,
                                target_dataset + target_subject + "_" + str(split_number))
            model.save_encoder_regressor(file)

    # save_file = os.path.join(save_file, "pretraining")
    results = ResultsSource(save_file, source_dataset, target_dataset, target_subject, model_name,
                            results=results)
    printd(results.get_results())

    if save_file is not None:
        results.save()


def main_target(mode, source_dataset, target_dataset, target_subject, Model, params, weights_dir, eval, split,
                save_file, plot):
    data = target_preprocessing(target_dataset, target_subject)

    splits = [split] if split is not None else range(cv * (cv - 1))
    results = []
    for split_number in splits:
        train, valid, test, means, stds = data.get_split(split_number)

        if mode in ["target_global", "target_finetuning"] and weights_dir is not None:
            weights_files = [os.path.join(domain_2_domain_str(source_dataset, target_dataset), weights_dir,
                                          Model.__name__ + "_" + target_dataset + target_subject + "_" + str(
                                              split) + ".pt") for split in range(cv)]
        else:
            weights_files = [None]

        for weights_file in weights_files:

            model = Model(params)
            if weights_file is not None: model.load_weights_from_file(weights_file)

            if not mode == "target_global":
                model.fit(*train, *valid)

            y_true, y_pred = model.predict(*test) if eval == "test" else model.predict(*valid)

            results.append(target_postprocessing(y_true, y_pred, means, stds))

    if mode == "target_global":
        suffix = "global"
    elif mode == "target_training":
        suffix = "training"
    elif mode == "target_finetuning":
        suffix = "finetuning"
    # save_file = os.path.join(save_file, suffix)
    save_file = save_file + "_" + suffix

    results = ResultsTarget(save_file, source_dataset, target_dataset, target_subject, model_name,
                            results=results)
    printd(results.get_results())

    if save_file is not None: results.save()

    printd(ResultsTarget(save_file, source_dataset, target_dataset, target_subject, model_name).get_results())

    if plot is not None and plot: results.plot()


if __name__ == "__main__":
    """
        --mode=source_training --source_dataset=IDIAB --target_dataset=IDIAB --target_subject=1 --model=DAFCN --eval=valid --save=test
        --mode=target_global --source_dataset=IDIAB --target_dataset=IDIAB --target_subject=1 --model=FCN --eval=valid --weights=test --save=test
        --mode=target_finetuning --source_dataset=IDIAB --target_dataset=IDIAB --target_subject=1 --model=FCN --eval=valid --weights=test --save=test
        --mode=target_training --source_dataset=IDIAB --target_dataset=IDIAB --target_subject=1 --model=FCN --eval=valid --save=test
    
        --mode: 4 modes 
                "source_training":      train a model on source dataset minus the target subject
                "target_training":      train a model on the target subject only
                "target_global":        use a model trained with the "source_training" mode to make the prediction for the 
                                        target subject. --weights_file must be set.
                "target_finetuning":    finetune a model trained with the "source_training" mode on the target subject
        --source_dataset:
                dataset used in the "source_training" mode, can be either "IDIAB", "Ohio" or "all"
        --target_dataset and --target_subject:
                specify the subject used in the "target_X" modes and removed from the "source_training" if needed
        --model:
                specify the model used in all the modes
        --weights:
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
    parser.add_argument("--weights", type=str)
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

    # create save directories
    if args.save_file is not None:
        dir = os.path.join(path, "models", "weights", domain_2_domain_str(args.source_dataset, args.target_dataset),
                           args.save_file)
        if not os.path.exists(dir): os.makedirs(dir)
        # os.makedirs(os.path.join(path, "results", args.save))

    if args.mode == "source_training":
        main_source(args.source_dataset, args.target_dataset, args.target_subject, Model, params, args.eval, args.split,
                    args.save_file)
    elif args.mode in ["target_training", "target_global", "target_finetuning"]:
        main_target(args.mode, args.source_dataset, args.target_dataset, args.target_subject, Model, params,
                    args.weights, args.eval,
                    args.split, args.save_file, args.plot)
