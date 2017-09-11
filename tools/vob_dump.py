# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 09/10/2017
#
# Description: wocabulary dump
#
# Last Modified at: 09/10/2017, by: Synrey Yee

import cPickle
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
      "--vecpath",
      type = str,
      default = "char_vec.txt",
      help = "vector's file")
    parser.add_argument(
      "--dump_path",
      type = str,
      default = "Models/vob_dump.pk",
      help = "dump path")

    args = parser.parse_args()

    vob = open(args.vecpath, 'r')
    lines = vob.readlines()
    first_line = lines[0].strip()
    words_num = int(first_line.split()[0])

    vob_dict = {}
    for i in xrange(words_num):
        line = lines[i + 1].strip()
        word = line.split()[0].decode("utf-8")
        vob_dict[word] = i

    vob.close()

    with open(args.dump_path, 'w') as dop:
        cPickle.dump(vob_dict, dop)