from misc.utils import locate_model, locate_params, printd
import sys
import argparse
import os
from misc.constants import *
from preprocessing.preprocessing import preprocessing, preprocessing_source_multi
from processing.cross_validation import make_predictions_tl
from postprocessing.postprocessing import postprocessing
from postprocessing.results import ResultsSubject

def main_target_training(source_dataset, target_dataset, target_subject, Model, params, weights_exp, eval_mode, exp, plot):
    hist_f = params["hist"] // freq
    weights_dir = os.path.join(path, "processing", "models", "weights", source_dataset + "_2_" + target_dataset,
                               weights_exp)
    weights_file = None
    save_file = None
    train, valid, test, scalers = preprocessing(target_dataset, target_subject, ph_f, hist_f, day_len_f)
    raw_results = make_predictions_tl(target_subject, Model, params, ph_f, train, valid, test, weights_file=weights_file,
                                      tl_mode="target_training", save_file=save_file, eval_mode=eval_mode)

    evaluation(raw_results, scalers, source_dataset, target_dataset, target_subject, Model, params, exp, plot, "target_training")


def main_source_training(source_dataset, target_dataset, target_subject, Model, params, weights_exp, eval_mode, exp, plot):
    hist_f = params["hist"] // freq
    weights_dir = os.path.join(path, "processing", "models", "weights", source_dataset + "_2_" + target_dataset,
                               weights_exp)
    save_file = os.path.join(weights_dir, Model.__name__ + "_" + target_dataset + target_subject + ".pt")
    weights_file = None
    train, valid, test, scalers = preprocessing_source_multi(source_dataset, target_dataset, target_subject, ph_f,
                                                             hist_f, day_len_f)
    raw_results = make_predictions_tl(target_subject, Model, params, ph_f, train, valid, test, weights_file=weights_file,
                                      tl_mode="source_training", save_file=save_file, eval_mode=eval_mode)



def main_target_global(source_dataset, target_dataset, target_subject, Model, params, weights_exp, eval_mode, exp, plot):
    hist_f = params["hist"] // freq
    weights_dir = os.path.join(path, "processing", "models", "weights", source_dataset + "_2_" + target_dataset,
                               weights_exp)
    weights_file = os.path.join(weights_dir, Model.__name__ + "_" + target_dataset + target_subject + ".pt")
    save_file = None
    train, valid, test, scalers = preprocessing(target_dataset, target_subject, ph_f, hist_f, day_len_f)

    raw_results = make_predictions_tl(target_subject, Model, params, ph_f, train, valid, test, weights_file=weights_file,
                                      tl_mode="target_global", save_file=save_file, eval_mode=eval_mode)

    evaluation(raw_results, scalers, source_dataset, target_dataset, target_subject, Model, params, exp, plot, "target_global")


def main_target_finetuning(source_dataset, target_dataset, target_subject, Model, params, weights_exp, eval_mode, exp, plot):
    hist_f = params["hist"] // freq
    weights_dir = os.path.join(path, "processing", "models", "weights", source_dataset + "_2_" + target_dataset,
                               weights_exp)
    weights_file = os.path.join(weights_dir, Model.__name__ + "_" + target_dataset + target_subject + ".pt")
    save_file = None
    train, valid, test, scalers = preprocessing(target_dataset, target_subject, ph_f, hist_f, day_len_f)

    raw_results = make_predictions_tl(target_subject, Model, params, ph_f, train, valid, test, weights_file=weights_file,
                                      tl_mode="target_finetuning", save_file=save_file, eval_mode=eval_mode)

    evaluation(raw_results, scalers, source_dataset, target_dataset, target_subject, Model, params, exp, plot, "target_finetuning")



def evaluation(raw_results, scalers, source_dataset, target_dataset, target_subject, Model, params, exp, plot, tl_mode):
    raw_results = postprocessing(raw_results, scalers, target_dataset)

    exp += "_" + tl_mode.split("_")[1]
    exp = os.path.join(source_dataset + "_2_" + target_dataset, exp)
    results = ResultsSubject(Model.__name__, exp, ph, target_dataset, target_subject, params=params,
                             results=raw_results)

    printd(results.compute_results())
    if plot:
        results.plot(0)


def process_main_args(args):
    Model = locate_model(args.model)
    params = locate_params(args.params)

    # redirect the logs to a file if specified
    if args.log is not None:
        log_file = args.log
        log_path = os.path.join(path, "logs", log_file)
        sys.stdout = open(log_path, "w")

    sbj_msg = args.source_dataset + "_2_" + args.target_dataset, " " + args.target_subject
    if args.tl_mode == "source_training":
        printd("source_training", sbj_msg)
        main_source_training(args.source_dataset, args.target_dataset, args.target_subject, Model, params,
                             args.weights, args.eval_mode, args.exp, args.plot)
    elif args.tl_mode == "target_training":
        printd("target_training", sbj_msg)
        main_target_training(args.source_dataset, args.target_dataset, args.target_subject, Model, params,
                             args.weights, args.eval_mode, args.exp, args.plot)
    elif args.tl_mode == "target_global":
        printd("target_global", sbj_msg)
        main_target_global(args.source_dataset, args.target_dataset, args.target_subject, Model, params,
                           args.weights, args.eval_mode, args.exp, args.plot)
    elif args.tl_mode == "target_finetuning":
        printd("target_finetuning", sbj_msg)
        main_target_finetuning(args.source_dataset, args.target_dataset, args.target_subject, Model, params,
                               args.weights, args.eval_mode, args.exp, args.plot)
    elif args.tl_mode == "end_to_end" and args.params2 is not None:
        printd("end_to_end", sbj_msg)

        params2 = locate_params(args.params2)

        main_source_training(args.source_dataset, args.target_dataset, args.target_subject, Model, params,
                             args.weights, args.eval_mode, args.exp, args.plot)
        main_target_global(args.source_dataset, args.target_dataset, args.target_subject, Model, params2,
                           args.weights, args.eval_mode, args.exp, args.plot)
        main_target_finetuning(args.source_dataset, args.target_dataset, args.target_subject, Model, params2,
                               args.weights, args.eval_mode, args.exp, args.plot)

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
    parser.add_argument("--params", type=str)
    parser.add_argument("--params2", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--eval_mode", type=str)
    parser.add_argument("--split", type=int)
    parser.add_argument("--log", type=str)
    parser.add_argument("--exp", type=str)
    parser.add_argument("--plot", type=bool)
    args = parser.parse_args()

    process_main_args(args)

