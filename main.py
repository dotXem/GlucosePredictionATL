from pydoc import locate
import sys
import argparse
import os
# from _misc import path, cv, freq
from preprocessing_old.preprocessing import source_preprocessing, target_preprocessing
from postprocessing_old.postprocessing import source_postprocessing, target_postprocessing
from evaluation_old.results import ResultsSource, ResultsTarget
from tools.printd import printd
from tools.domain2domain import domain_2_domain_str

from misc.constants import *
from preprocessing.preprocessing import preprocessing
from processing.cross_validation import make_predictions
from postprocessing.postprocessing import postprocessing
from postprocessing.results import ResultsSubject


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

        model.clear_checkpoint()

    results = ResultsSource(save_file, source_dataset, target_dataset, target_subject, Model.__name__,
                            results=results)
    printd(results.get_results())

    if save_file is not None:
        results.save()


def main_target(tl_mode, source_dataset, target_dataset, target_subject, Model, params, eval_mode, exp, plot):
    hist_f = params["hist"] // freq

    if tl_mode in ["target_global", "target_finetuning"]:
        weights_files = [os.path.join(path, "processing", "models", "weights", source_dataset + "_2_" + target_dataset,
                                      Model.__name__ + "_" + target_dataset + target_subject + "_" + str(
                                          split) + ".pt") for split in range(cv)]
    else:
        weights_files = [None]

    train, valid, test, scalers = preprocessing(target_dataset, target_subject, ph_f, hist_f, day_len_f)

    raw_results = make_predictions(target_subject, Model, params, ph_f, train, valid, test, weights_files=weights_files,
                                   tl_mode=tl_mode, eval_mode=eval_mode)
    raw_results = postprocessing(raw_results, scalers, target_dataset)

    exp += "_" + tl_mode.split("_")[1]
    results = ResultsSubject(Model.__name__, exp, ph, target_dataset, target_subject, params=params,
                             results=raw_results)

    printd(results.compute_results())
    if plot:
        results.plot(0)


# def main_target(mode, source_dataset, target_dataset, target_subject, Model, params, weights_dir, eval, split,
#                 save_file, plot):

# data = target_preprocessing(target_dataset, target_subject)
#
# splits = [split] if split is not None else range(cv * (cv - 1))
# results = []
# for split_number in splits:
#     train, valid, test, means, stds = data.get_split(split_number)
#
#     if mode in ["target_global", "target_finetuning"] and weights_dir is not None:
#         weights_files = [os.path.join(domain_2_domain_str(source_dataset, target_dataset), weights_dir,
#                                       Model.__name__ + "_" + target_dataset + target_subject + "_" + str(
#                                           split) + ".pt") for split in range(cv)]
#     else:
#         weights_files = [None]
#
#     for weights_file in weights_files:
#
#         model = Model(params)
#         if weights_file is not None: model.load_weights_from_file(weights_file)
#
#         if not mode == "target_global":
#             model.fit(*train, *valid)
#
#         y_true, y_pred = model.predict(*test) if eval == "test" else model.predict(*valid)
#
#         results.append(target_postprocessing(y_true, y_pred, means, stds))
#
#         model.clear_checkpoint()
#
# if mode == "target_global":
#     suffix = "global"
# elif mode == "target_training":
#     suffix = "training_old"
# elif mode == "target_finetuning":
#     suffix = "finetuning"
# save_file = save_file + "_" + suffix
#
# results = ResultsTarget(save_file, source_dataset, target_dataset, target_subject, Model.__name__,
#                         results=results)
# printd(results.get_results())
#
# if save_file is not None: results.save()
#
# if plot is not None and plot: results.plot()


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
                specify the evaluation_old set to be used, in the "target_X" modes, either "valid" or "test". default:
                "valid".
        --split:
                restrict the training_old/evaluation_old to one specific split in [0-11] for "traget_X" modes and [0-3] in 
                "source_training" mode. default: None
        --log:
                specify the file where the logs shall be redirected to. default: None
        --save:
                if set, saves the prediction results in the specified folder ("target_X" modes) and the model checkpoints
                in the "source_training_mode". default: False
        --plot:
                if set, plot the results after the training_old. default: True  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tl_mode", type=str)
    parser.add_argument("--source_dataset", type=str)
    parser.add_argument("--target_dataset", type=str)
    parser.add_argument("--target_subject", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--eval_mode", type=str)
    parser.add_argument("--split", type=int)
    parser.add_argument("--log", type=str)
    parser.add_argument("--exp", type=str)
    parser.add_argument("--plot", type=bool)
    args = parser.parse_args()

    model_name = args.model if args.model is not None else sys.exit(-1)
    Model = locate("processing.models." + model_name + "." + model_name.upper())
    params = locate("processing.params." + model_name + ".parameters")

    # redirect the logs to a file if specified
    if args.log is not None:
        log_file = args.log
        log_path = os.path.join(path, "logs", log_file)
        sys.stdout = open(log_path, "w")

    # create save directories
    if args.exp is not None:
        dir = os.path.join(path, "models", "weights", domain_2_domain_str(args.source_dataset, args.target_dataset),
                           args.exp)
        if not os.path.exists(dir): os.makedirs(dir)
        # os.makedirs(os.path.join(path, "results", args.save))

    if args.tl_mode == "source_training":
        main_source(args.source_dataset, args.target_dataset, args.target_subject, Model, params, args.eval, args.split,
                    args.save_file)
    elif args.tl_mode in ["target_training", "target_global", "target_finetuning"]:
        # main_target(args.mode, args.source_dataset, args.target_dataset, args.target_subject, Model, params,
        #             args.weights, args.eval, args.split, args.save_file, args.plot)
        main_target(args.tl_mode, args.source_dataset, args.target_dataset, args.target_subject, Model, params,
                    args.eval_mode, args.exp, args.plot)
