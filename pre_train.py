# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 05/07/2017
#
# Description: generate training relevant files
#
# Last Modified at: 07/10/2017, by: Synrey Yee


import SentHandler as snhd
import argparse

def main(corpusAll,
         vecpath,
         train_file,
         test_file,
         test_file_raw,
         test_file_gold,
         MAX_LEN,
         test_size,
         step):

    vob = open(vecpath, 'r')
    lines = vob.readlines()
    first_line = lines[0].strip()
    words_num = int(first_line.split()[0])

    vob_dict = {}
    for i in xrange(words_num):
        line = lines[i + 1].strip()
        word = line.split()[0].decode("utf-8")
        vob_dict[word] = i

    vob.close()

    inp = open(corpusAll, 'r')
    bad_lines = 0
    cnt = 0

    trop_ = open(train_file, 'w')
    ttop_ = open(test_file, 'w')
    ttrop_ = open(test_file_raw, 'w')
    ttgop_ = open(test_file_gold, 'w')
    with trop_ as trop, ttop_ as ttop, ttrop_ as ttrop, ttgop_ as ttgop:
        for ind, line in enumerate(inp):
            line_pieces = []
            NE_free_line = snhd.NE_Removing(line)
            flag = snhd.SliceSentence(NE_free_line, line_pieces, [], tag = True, max_len = MAX_LEN)

            if not flag:
                bad_lines += 1
                continue

            analyzed_pieces, flag = snhd.Analyze(line_pieces, vob_dict, max_len = MAX_LEN)

            if not flag:
                bad_lines += 1
                continue

            if (ind + 1) % step == 0 and cnt < test_size :
                for piece_raw, piece in zip(line_pieces, analyzed_pieces):
                    ttrop.write(snhd.CleanSentence(piece_raw, set([])))
                    ttgop.write(snhd.CleanSentence(piece_raw, set([]), interval = u' '))
                    ttop.write(piece)
                cnt += 1
            else:
                for piece in analyzed_pieces:
                    trop.write(piece)

        print "Generating finished, gave up %d bad lines" % bad_lines

    inp.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
      "--corpusAll",
      type = str,
      default = "Corpora/people2014All.txt",
      help = "corpus file")
    parser.add_argument(
      "--vecpath",
      type = str,
      default = "char_vec.txt",
      help = "vector's file")
    parser.add_argument(
      "--train_file",
      type = str,
      default = "Corpora/train.txt",
      help = "training file will be generated here")
    parser.add_argument(
      "--test_file",
      type = str,
      default = "Corpora/test.txt",
      help = "testing file will be generated here")
    parser.add_argument(
      "--test_file_raw",
      type = str,
      default = "Corpora/test_raw.txt",
      help = "testing raw file will be generated here")
    parser.add_argument(
      "--test_file_gold",
      type = str,
      default = "Corpora/test_gold.txt",
      help = "gold file will be generated here")

    parser.add_argument(
      "--MAX_LEN",
      type = int,
      default = 80,
      help = "max sentencen length")
    parser.add_argument(
      "--test_size",
      type = int,
      default = 8000,
      help = "the sentence lines of testing file")
    parser.add_argument(
      "--step",
      type = int,
      default = 50,
      help = "program chooses 1 test sentence for every <step> steps")

    args = parser.parse_args()
    main(args.corpusAll, args.vecpath, args.train_file,
        args.test_file, args.test_file_raw, args.test_file_gold,
        args.MAX_LEN, args.test_size, args.step)