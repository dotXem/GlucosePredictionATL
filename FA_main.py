from misc.utils import printd
from postprocessing.features_analysis import FeaturesAnalyzer
import argparse
import sys
import os
import misc.constants


def main(metric, source, target, exp, model, params, neighbours, to_other, use_tsne, save):
    params_str = metric + " " +  source +  " " +  target + " " + exp + " " + model + " " + str(neighbours) + " " + str(to_other) + " " + str(use_tsne) + " "

    save_file = source + "_2_" + target + "_" + exp + "_" + metric + ".npy"
    save_file = os.path.join(misc.constants.path, "results", "features_analysis", save_file) if save is not None else None

    FA = FeaturesAnalyzer(source, target, exp, model, params)

    if metric == "perplexity":
        res = FA.perplexity(neighbours, use_tsne=bool(use_tsne), save_file=save_file)
    elif metric == "distance":
        res = FA.distance(to_other=bool(to_other), use_tsne=bool(use_tsne), save_file=save_file)


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

    main(args.metric, args.source, args.target, args.exp, args.model, args.params, args.neighbours, args.to_other, args.use_tsne,
         args.save)

    # for source, target in [["t1dms","idiab"],["idiab+ohio","idiab"]]:
    #     main(args.metric, source, target, args.exp, args.model, args.params, args.neighbours, args.to_other, args.use_tsne, args.save)


    # f = open(args.log_file, 'a+')
    # sys.stdout = f
    #
    # printd(params_str, res)
    # f.close()
