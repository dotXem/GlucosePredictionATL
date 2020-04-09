from misc.utils import printd
from postprocessing.features_analysis import FeaturesAnalyzer
import argparse
import sys
import os
import misc.constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--exp", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--params", type=str)
    parser.add_argument("--metric", type=str)
    parser.add_argument("--neighbours", type=int)
    parser.add_argument("--use_tsne", type=int)
    parser.add_argument("--to_other", type=int)
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--save", type=int)
    args = parser.parse_args()
    
    params_str = args.metric + " " +  args.source +  " " +  args.target + " " + args.exp + " " + args.model + " " + str(args.neighbours) + " " + str(args.to_other) + " " + str(args.use_tsne) + " " 

    save_file = args.source + "_2_" + args.target + "_" + args.exp + "_" + args.metric + ".npy"
    save_file = os.path.join(misc.constants.path, "results", "features_analysis", save_file) if args.save is not None else None

    FA = FeaturesAnalyzer(args.source, args.target, args.exp, args.model, args.params)

    if args.metric == "perplexity":
        res = FA.perplexity(args.neighbours, use_tsne=bool(args.use_tsne), save_file=save_file)
    elif args.metric == "distance":
        res = FA.distance(to_other=bool(args.to_other), use_tsne=bool(args.use_tsne), save_file=save_file)



    # f = open(args.log_file, 'a+')
    # sys.stdout = f
    #
    # printd(params_str, res)
    # f.close()
