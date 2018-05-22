# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 03/21/2018
#
# Description: A neural network tool for Chinese word segmentation
#
# Last Modified at: 05/21/2018, by: Synrey Yee

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

from __future__ import absolute_import
from __future__ import print_function

from . import main_body
from . import model as model_r

import tensorflow as tf

import argparse
import sys
import os
import codecs


# the parsed parameters
FLAGS = None
UNK = u"unk"


# build the training parameters
def add_arguments(parser):
  parser.register("type", "bool", lambda v: v.lower() == "true")

  # data
  parser.add_argument("--train_prefix", type = str, default = None,
                      help = "Train prefix, expect files with .txt/.lb suffixes.")
  parser.add_argument("--eval_prefix", type = str, default = None,
                      help = "Eval prefix, expect files with .txt/.lb suffixes.")
  parser.add_argument("--eval_gold_file", type = str, default = None,
                      help = "Eval gold file.")

  parser.add_argument("--vocab_file", type = str, default = None,
                      help = "Vocabulary file.")
  parser.add_argument("--embed_file", type = str, default = None, help="""\
      Pretrained embedding files, The expecting files should be Glove formated txt files.\
      """)
  parser.add_argument("--index_file", type = str, default = "./sycws/indices.txt",
                      help = "Indices file.")
  parser.add_argument("--out_dir", type = str, default = None,
                      help = "Store log/model files.")
  parser.add_argument("--max_len", type = int, default = 150,
                      help = "Max length of char sequences during training.")

  # hyperparameters
  parser.add_argument("--num_units", type = int, default = 100, help = "Network size.")
  parser.add_argument("--model", type = str, default = "CNN-CRF",
      help = "2 kind of models: BiLSTM + (CRF | CNN-CRF)")

  parser.add_argument("--learning_rate", type = float, default = 0.001,
                      help = "Learning rate. Adam: 0.001 | 0.0001")

  parser.add_argument("--num_train_steps",
      type = int, default = 45000, help = "Num steps to train.")

  parser.add_argument("--init_std", type = float, default = 0.05,
                      help = "for truncated normal init_op")
  parser.add_argument("--filter_init_std", type = float, default = 0.035,
                      help = "truncated normal initialization for CNN's filter")

  parser.add_argument("--dropout", type = float, default = 0.3,
                      help = "Dropout rate (not keep_prob)")
  parser.add_argument("--max_gradient_norm", type = float, default = 5.0,
                      help = "Clip gradients to this norm.")
  parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size.")

  parser.add_argument("--steps_per_stats", type = int, default = 100,
                      help = "How many training steps to print loss.")
  parser.add_argument("--num_buckets", type = int, default = 5,
                      help = "Put data into similar-length buckets.")
  parser.add_argument("--steps_per_external_eval", type = int, default = None,
                      help = """\
      How many training steps to do per external evaluation.  Automatically set
      based on data if None.\
      """)

  parser.add_argument("--num_tags", type = int, default = 4, help = "BMES")
  parser.add_argument("--time_major", type = "bool", nargs = "?", const = True,
                      default = True,
                      help = "Whether to use time_major mode for dynamic RNN.")

  # Inference
  parser.add_argument("--ckpt", type = str, default = None,
                      help = "Checkpoint file to load a model for inference.")
  parser.add_argument("--inference_input_file", type = str, default = None,
                      help = "Set to the text to decode.")
  parser.add_argument("--infer_batch_size", type = int, default = 32,
                      help = "Batch size for inference mode.")
  parser.add_argument("--inference_output_file", type = str, default = None,
                      help = "Output file to store decoding results.")

def create_hparams(flags):
  """Create training hparams."""
  return tf.contrib.training.HParams(
    # data
    train_prefix = flags.train_prefix,
    eval_prefix = flags.eval_prefix,
    eval_gold_file = flags.eval_gold_file,
    vocab_file = flags.vocab_file,
    embed_file = flags.embed_file,

    index_file = flags.index_file,
    out_dir = flags.out_dir,
    max_len = flags.max_len,

    # hparams
    num_units = flags.num_units,
    model = flags.model,
    learning_rate = flags.learning_rate,
    num_train_steps = flags.num_train_steps,

    init_std = flags.init_std,
    filter_init_std = flags.filter_init_std,
    dropout = flags.dropout,
    max_gradient_norm = flags.max_gradient_norm,

    batch_size = flags.batch_size,
    num_buckets = flags.num_buckets,
    steps_per_stats = flags.steps_per_stats,
    steps_per_external_eval = flags.steps_per_external_eval,

    num_tags = flags.num_tags,
    time_major = flags.time_major,

    # inference
    ckpt = flags.ckpt,
    inference_input_file = flags.inference_input_file,
    infer_batch_size = flags.infer_batch_size,
    inference_output_file = flags.inference_output_file,
    )


def check_corpora(train_prefix, eval_prefix):
  train_txt = train_prefix + ".txt"
  train_lb = train_prefix + ".lb"
  eval_txt = eval_prefix + ".txt"
  eval_lb = eval_prefix + ".lb"

  def _inner_check(txt_reader, lb_reader):
    for txt_line, lb_line in zip(txt_reader, lb_reader):
      txt_length = len(txt_line.strip().split())
      lb_length = len(lb_line.strip().split())
      assert txt_length == lb_length

  train_txt_rd = codecs.open(train_txt, 'r', "utf-8")
  train_lb_rd = codecs.open(train_lb, 'r', "utf-8")
  eval_txt_rd = codecs.open(eval_txt, 'r', "utf-8")
  eval_lb_rd = codecs.open(eval_lb, 'r', "utf-8")

  with train_txt_rd, train_lb_rd:
    _inner_check(train_txt_rd, train_lb_rd)

  with eval_txt_rd, eval_lb_rd:
    _inner_check(eval_txt_rd, eval_lb_rd)


def check_vocab(vocab_file):
  vocab = []
  with codecs.open(vocab_file, 'r', "utf-8") as vob_inp:
    for word in vob_inp:
      vocab.append(word.strip())

  if vocab[0] != UNK:
    vocab = [UNK] + vocab
    with codecs.open(vocab_file, 'w', "utf-8") as vob_opt:
      for word in vocab:
        vob_opt.write(word + u'\n')

  return len(vocab)


def print_hparams(hparams):
  values = hparams.values()
  for key in sorted(values.keys()):
    print("  %s = %s" % (key, str(values[key])))


def main(unused_argv):
  out_dir = FLAGS.out_dir
  if not tf.gfile.Exists(out_dir):
    tf.gfile.MakeDirs(out_dir)

  hparams = create_hparams(FLAGS)
  model = hparams.model.upper()
  if model == "CRF":
    model_creator = model_r.BasicModel
  elif model == "CNN-CRF":
    model_creator = model_r.CnnCrfModel
  else:
    raise ValueError("Unknown model %s" % model)

  assert tf.gfile.Exists(hparams.vocab_file)
  vocab_size = check_vocab(hparams.vocab_file)
  hparams.add_hparam("vocab_size", vocab_size)

  if FLAGS.inference_input_file:
    # Inference
    trans_file = FLAGS.inference_output_file
    ckpt = FLAGS.ckpt
    if not ckpt:
      ckpt = tf.train.latest_checkpoint(out_dir)

    main_body.inference(ckpt, FLAGS.inference_input_file,
                 trans_file, hparams, model_creator)
  else:
    # Train
    check_corpora(FLAGS.train_prefix, FLAGS.eval_prefix)

    # used for evaluation
    hparams.add_hparam("best_Fvalue", 0)  # larger is better
    best_metric_dir = os.path.join(hparams.out_dir, "best_Fvalue")
    hparams.add_hparam("best_Fvalue_dir", best_metric_dir)
    tf.gfile.MakeDirs(best_metric_dir)

    print_hparams(hparams)
    main_body.train(hparams, model_creator)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)