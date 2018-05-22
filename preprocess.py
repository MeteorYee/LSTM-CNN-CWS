# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 03/22/2018
#
# Description: preprocessing for people 2014 corpora
#
# Last Modified at: 05/20/2018, by: Synrey Yee

'''
==========================================================================
  Copyright 2018 Xingyu Yi (Alias: Synrey Yee) All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
==========================================================================
'''

from __future__ import print_function

from collections import defaultdict

import codecs
import argparse
import os


NE_LEFT = u'['
NE_RIGHT = u']'
DIVIDER = u'/'
SPACE = u' '
UNK = u"unk"

WORD_S = u'0'
WORD_B = u'1'
WORD_M = u'2'
WORD_E = u'3'


def clean_sentence(line_list):
  new_line_list = []
  for token in line_list:
    div_id = token.rfind(DIVIDER)
    
    if div_id < 1:
      # the div_id shouldn't be lower than 1
      # if it does, give up the word
      continue

    word = token[ : div_id]
    tag = token[(div_id + 1) : ]

    if word[0] == NE_LEFT and len(word) > 1:
      new_line_list.append(word[1 : ])

    elif word[-1] == NE_RIGHT and len(word) > 1:
      div_id = word.rfind(DIVIDER)
      if div_id < 1:
        new_line_list.append(word[ : (len(word)-1)])
      else:
        new_line_list.append(word[ : div_id])

    else:
      new_line_list.append(word)

  return new_line_list


def write_line(line_list, outstream, sep = SPACE):
  line = sep.join(line_list)
  outstream.write(line + u'\n')


def analyze_line(line_list, vob_dict):
  char_list = []
  label_list = []

  for word in line_list:
    length = len(word)
    if length == 1:
      char_list.append(word)
      label_list.append(WORD_S)
      vob_dict[word] += 1

    else:
      for pos, char in enumerate(word):
        if pos == 0:
          label_list.append(WORD_B)
        elif pos == (length - 1):
          label_list.append(WORD_E)
        else:
          label_list.append(WORD_M)

        char_list.append(char)
        vob_dict[char] += 1

  assert len(char_list) == len(label_list)
  return char_list, label_list


def generate_files(corpora, vob_path, char_file, train_word_file,
    train_label_file, eval_word_file, eval_label_file, eval_gold_file,
    test_file, gold_file, step, freq, max_len):

  inp = codecs.open(corpora, 'r', "utf-8")

  tr_wd_wr = codecs.open(train_word_file, 'w', "utf-8")
  tr_lb_wr = codecs.open(train_label_file, 'w', "utf-8")
  ev_wd_wr = codecs.open(eval_word_file, 'w', "utf-8")
  ev_lb_wr = codecs.open(eval_label_file, 'w', "utf-8")

  ev_gold_wr = codecs.open(eval_gold_file, 'w', "utf-8")
  test_wr = codecs.open(test_file, 'w', "utf-8")
  gold_wr = codecs.open(gold_file, 'w', "utf-8")

  dump_cnt = 0
  vob_dict = defaultdict(int)
  isEval = True

  with inp, tr_wd_wr, tr_lb_wr, ev_wd_wr, ev_lb_wr, ev_gold_wr, test_wr, gold_wr:
    for ind, line in enumerate(inp):
      line_list = line.strip().split()
      if len(line_list) > max_len:
        dump_cnt += 1
        continue

      cleaned_line = clean_sentence(line_list)
      if not cleaned_line:
        dump_cnt += 1
        continue

      char_list, label_list = analyze_line(cleaned_line, vob_dict)

      if ind % step == 0:
        if isEval:
          write_line(char_list, ev_wd_wr)
          write_line(label_list, ev_lb_wr)
          write_line(cleaned_line, ev_gold_wr)
          isEval = False
        else:
          write_line(cleaned_line, test_wr, sep = u'')
          write_line(cleaned_line, gold_wr)
          isEval = True
      else:
        write_line(char_list, tr_wd_wr)
        write_line(label_list, tr_lb_wr)

  inp = codecs.open(corpora, 'r', "utf-8")
  ch_wr = codecs.open(char_file, 'w', "utf-8")
  with inp, ch_wr:
    for line in inp:
      line_list = line.strip().split()
      if len(line_list) > max_len:
        continue

      cleaned_line = clean_sentence(line_list)
      if not cleaned_line:
        continue
      char_list = []
      for phr in cleaned_line:
        for ch in phr:
          if vob_dict[ch] < freq:
            char_list.append(UNK)
          else:
            char_list.append(ch)

      write_line(char_list, ch_wr)

  word_cnt = 0
  with codecs.open(vob_path, 'w', "utf-8") as vob_wr:
    vob_wr.write(UNK + u'\n')
    for word, fq in vob_dict.items():
      if fq >= freq:
        vob_wr.write(word + u'\n')
        word_cnt += 1

  print("Finished, give up %d sentences." % dump_cnt)
  print("Select %d chars from the original %d chars" % (word_cnt, len(vob_dict)))


# used for people corpora
def people_main(args):
  corpora = args.all_corpora
  assert os.path.exists(corpora)

  total_line = 0
  # count the total number of lines
  with open(corpora, 'rb') as inp:
    for line in inp:
      total_line += 1

  base = 2 * args.line_cnt
  assert base < total_line
  step = total_line // base

  train_word_file = args.train_file_pre + ".txt"
  train_label_file = args.train_file_pre + ".lb"
  eval_word_file = args.eval_file_pre + ".txt"
  eval_label_file = args.eval_file_pre + ".lb"

  generate_files(corpora, args.vob_path, args.char_file,
      train_word_file, train_label_file, eval_word_file,
      eval_label_file, args.eval_gold_file, args.test_file,
      args.gold_file, step, args.word_freq, args.max_len)


def analyze_write(inp, word_writer, label_writer,
    vob_dict = defaultdict(int)):
  with inp, word_writer, label_writer:
    for line in inp:
      line_list = line.strip().split()
      if len(line_list) < 1:
        continue

      char_list, label_list = analyze_line(line_list, vob_dict)
      write_line(char_list, word_writer)
      write_line(label_list, label_writer)


# used for icwb2 data
def icwb_main(args):
  corpora = args.all_corpora
  assert os.path.exists(corpora)
  gold_file = args.gold_file
  assert os.path.exists(gold_file)
  freq = args.word_freq

  train_word_file = args.train_file_pre + ".txt"
  train_label_file = args.train_file_pre + ".lb"
  eval_word_file = args.eval_file_pre + ".txt"
  eval_label_file = args.eval_file_pre + ".lb"

  train_inp = codecs.open(corpora, 'r', "utf-8")
  gold_inp = codecs.open(gold_file, 'r', "utf-8")

  ch_wr = codecs.open(args.char_file, 'w', "utf-8")
  tr_wd_wr = codecs.open(train_word_file, 'w', "utf-8")
  tr_lb_wr = codecs.open(train_label_file, 'w', "utf-8")
  ev_wd_wr = codecs.open(eval_word_file, 'w', "utf-8")
  ev_lb_wr = codecs.open(eval_label_file, 'w', "utf-8")

  vob_dict = defaultdict(int)
  analyze_write(train_inp, tr_wd_wr, tr_lb_wr, vob_dict)
  analyze_write(gold_inp, ev_wd_wr, ev_lb_wr)

  train_inp = codecs.open(corpora, 'r', "utf-8")
  with train_inp, ch_wr:
    for line in train_inp:
      phrases = line.strip().split()
      char_list = []
      for phr in phrases:
        for ch in phr:
          if vob_dict[ch] < freq:
            char_list.append(UNK)
          else:
            char_list.append(ch)

      write_line(char_list, ch_wr)

  word_cnt = 0
  with codecs.open(args.vob_path, 'w', "utf-8") as vob_wr:
    vob_wr.write(UNK + u'\n')
    for word, fq in vob_dict.items():
      if fq >= freq:
        vob_wr.write(word + u'\n')
        word_cnt += 1

  print("Finished, handling icwb2 data.")
  print("Select %d chars from the original %d chars" % (word_cnt, len(vob_dict)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")

  # input
  parser.add_argument(
    "--all_corpora",
    type = str,
    default = "/home/synrey/data/people2014All.txt",
    help = "all the corpora")

  # output
  parser.add_argument(
    "--vob_path",
    type = str,
    default = "/home/synrey/data/cws-v2-data/vocab.txt",
    help = "vocabulary's path")

  parser.add_argument(
    "--char_file",
    type = str,
    default = "/home/synrey/data/cws-v2-data/chars.txt",
    help = "the file used for word2vec pretraining")

  parser.add_argument(
    "--train_file_pre",
    type = str,
    default = "/home/synrey/data/cws-v2-data/train",
    help = "training file's prefix")

  parser.add_argument(
    "--eval_file_pre",
    type = str,
    default = "/home/synrey/data/cws-v2-data/eval",
    help = "eval file's prefix")

  parser.add_argument(
    "--eval_gold_file",
    type = str,
    default = "/home/synrey/data/cws-v2-data/eval_gold.txt",
    help = """gold file, used for the evaluation during training, \
      only generated for the 'people' corpus""")

  parser.add_argument(
    "--test_file",
    type = str,
    default = "/home/synrey/data/cws-v2-data/test.txt",
    help = "test file, raw sentences")

  parser.add_argument(
    "--gold_file",
    type = str,
    default = "/home/synrey/data/cws-v2-data/gold.txt",
    help = "gold file, segmented sentences")

  # parameters
  parser.add_argument(
    "--word_freq",
    type = int,
    default = 3,
    help = "word frequency")

  parser.add_argument(
    "--line_cnt",
    type = int,
    default = 8000,
    help = "the number of lines in eval or test file")

  # NOTE: It is the max length of word sequence, not char.
  parser.add_argument(
    "--max_len",
    type = int,
    default = 120,
    help = "deprecate the sentences longer than <max_len>")

  parser.add_argument(
    "--is_people",
    type = "bool",
    default = True,
    help = "Whether it is handling with People corpora")

  args = parser.parse_args()
  if args.is_people:
    people_main(args)
  else:
    icwb_main(args)