# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 04/24/2017
#
# Description: pre-processing
#
# Last Modified at: 05/28/2017, by: Synrey Yee

import SentHandler as snhd
import cPickle
import argparse
import os

def main(rootDir, corpusAll, resultFile, flag, pos_tags):
    if not os.path.exists(corpusAll):
        # make all the corpus files into one file
        snhd.All2oneFile(rootDir, corpusAll)

    # used to collect tags
    tags = set([])

    with open(resultFile, 'w') as opt:
        inp = open(corpusAll, 'r')
        for line in inp:
            NE_free_line = snhd.NE_Removing(line)
            newline = snhd.CleanSentence(NE_free_line, tags, interval = u' ', breakword = flag)
            opt.write(newline)

        inp.close()

    print "got %d tags" % len(tags)
    with open(pos_tags, 'w') as opt:
        cPickle.dump(tags, opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
      "--rootDir",
      type = str,
      default = "",
      help = "root directory of the corpus")
    parser.add_argument(
      "--corpusAll",
      type = str,
      default = "",
      help = "all corpus files will be saved into file <corpusAll>")
    parser.add_argument(
      "--resultFile",
      type = str,
      default = "pre_chars_for_w2v.txt",
      help = "the result file of preprocessing")
    parser.add_argument(
      "--flag",
      nargs = "?",
      const = True,
      type = "bool",
      default = True,
      help = "if want to preprocess NER or POS tagging, set flag to 'False'")
    parser.add_argument(
      "--pos_tags",
      type = str,
      default = "Models/pos_tags.cpk",
      help = "pos_tags will be saved in this file by cPickle")

    args = parser.parse_args()
    main(args.rootDir, args.corpusAll, args.resultFile, args.flag, args.pos_tags)