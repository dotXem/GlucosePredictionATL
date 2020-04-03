from main import process_main_args
from misc.datasets import datasets
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tl_mode", type=str)
    parser.add_argument("--source_dataset", type=str)
    parser.add_argument("--target_dataset", type=str)
    parser.add_argument("--target_subject",type=str)
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

    for subject in datasets[args.target_dataset]["subjects"]:
        args.target_subject = subject
        process_main_args(args)