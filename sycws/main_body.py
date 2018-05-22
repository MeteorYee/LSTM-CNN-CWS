# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 03/24/2018
#
# Description: Training, evaluation and inference
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
from __future__ import division

from . import model_helper
from . import prf_script

import tensorflow as tf
import numpy as np

import time
import os
import codecs


TAG_S = 0
TAG_B = 1
TAG_M = 2
TAG_E = 3


def train(hparams, model_creator):
  out_dir = hparams.out_dir
  num_train_steps = hparams.num_train_steps
  steps_per_stats = hparams.steps_per_stats
  steps_per_external_eval = hparams.steps_per_external_eval
  if not steps_per_external_eval:
    steps_per_external_eval = 10 * steps_per_stats

  train_model = model_helper.create_train_model(hparams, model_creator)
  eval_model = model_helper.create_eval_model(hparams, model_creator)
  infer_model = model_helper.create_infer_model(hparams, model_creator)

  eval_txt_file = "%s.%s" % (hparams.eval_prefix, "txt")
  eval_lb_file = "%s.%s" % (hparams.eval_prefix, "lb")
  eval_iterator_feed_dict = {
    eval_model.txt_file_placeholder : eval_txt_file,
    eval_model.lb_file_placeholder : eval_lb_file
  }

  model_dir = hparams.out_dir

  # TensorFlow model
  train_sess = tf.Session(graph = train_model.graph)
  eval_sess = tf.Session(graph = eval_model.graph)
  infer_sess = tf.Session(graph = infer_model.graph)

  # Read the infer data
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(eval_txt_file, mode="rb")) as f:
    infer_data = f.read().splitlines()

  with train_model.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, model_dir, train_sess, "train", init = True)

  print("First evaluation:")
  _run_full_eval(hparams, loaded_train_model, train_sess, eval_model,
      model_dir, eval_sess, eval_iterator_feed_dict, infer_model,
      infer_sess, infer_data, global_step, init = True)

  print("# Initialize train iterator...")
  train_sess.run(train_model.iterator.initializer)

  process_time = 0.0
  while global_step < num_train_steps:
    # train a batch
    start_time = time.time()
    try:
      step_result = loaded_train_model.train(train_sess)
      process_time += time.time() - start_time
    except tf.errors.OutOfRangeError:
      # finish one epoch
      print(
          "# Finished an epoch, step %d. Perform evaluation" %
          global_step)

      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "segmentation.ckpt"),
          global_step = global_step)
      _run_full_eval(hparams, loaded_train_model, train_sess, eval_model,
          model_dir, eval_sess, eval_iterator_feed_dict, infer_model,
          infer_sess, infer_data, global_step, init = False)

      train_sess.run(train_model.iterator.initializer)
      continue

    _, train_loss, global_step, batch_size = step_result
    if global_step % steps_per_stats == 0:
      avg_time = process_time / steps_per_stats
      # print loss info
      print("[%d][loss]: %f, time per step: %.2fs" % (global_step,
          train_loss, avg_time))
      process_time = 0.0

    if global_step % steps_per_external_eval == 0:
      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "segmentation.ckpt"),
          global_step = global_step)

      print("External Evaluation:")
      _run_full_eval(hparams, loaded_train_model, train_sess, eval_model,
          model_dir, eval_sess, eval_iterator_feed_dict, infer_model,
          infer_sess, infer_data, global_step, init = False)


def evaluation(eval_model, model_dir, eval_sess,
    eval_iterator_feed_dict, init = True):
  with eval_model.graph.as_default():
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, model_dir, eval_sess, "eval", init)

  eval_sess.run(eval_model.iterator.initializer,
      feed_dict = eval_iterator_feed_dict)

  total_char_cnt = 0
  total_right_cnt = 0
  total_line = 0
  while True:
    try:
      (batch_char_cnt, batch_right_cnt, batch_size,
          batch_lens) = loaded_eval_model.eval(eval_sess)

      for right_cnt, length in zip(batch_right_cnt, batch_lens):
        total_right_cnt += np.sum(right_cnt[ : length])
        
      total_char_cnt += batch_char_cnt
      total_line += batch_size
    except tf.errors.OutOfRangeError:
      # finish the evaluation
      break

  precision = total_right_cnt / total_char_cnt
  print("Tagging precision: %.3f, of total %d lines" % (precision, total_line))


def _eval_inference(infer_model, infer_sess, infer_data, model_dir, hparams, init):
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer", init)

  infer_sess.run(
        infer_model.iterator.initializer,
        feed_dict = {
            infer_model.txt_placeholder: infer_data,
            infer_model.batch_size_placeholder: hparams.infer_batch_size
        })

  test_list = []
  while True:
    try:
      text_raw, decoded_tags, seq_lens = loaded_infer_model.infer(infer_sess)
    except tf.errors.OutOfRangeError:
      # finish the evaluation
      break
    _decode_by_function(lambda x : test_list.append(x), text_raw, decoded_tags, seq_lens)

  gold_file = hparams.eval_gold_file
  score = prf_script.get_prf_score(test_list, gold_file)
  return score


def _run_full_eval(hparams, loaded_train_model, train_sess, eval_model,
    model_dir, eval_sess, eval_iterator_feed_dict, infer_model,
    infer_sess, infer_data, global_step, init):

  evaluation(eval_model, model_dir, eval_sess,
    eval_iterator_feed_dict)
  score = _eval_inference(infer_model, infer_sess, infer_data,
      model_dir, hparams, init)
  # save the best model
  if score > getattr(hparams, "best_Fvalue"):
    setattr(hparams, "best_Fvalue", score)
    loaded_train_model.saver.save(
        train_sess,
        os.path.join(
            getattr(hparams, "best_Fvalue_dir"), "translate.ckpt"),
        global_step = global_step)


def load_data(inference_input_file):
  # Load inference data.
  inference_data = []
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(inference_input_file, mode="rb")) as f:
    for line in f:
      line = line.strip()
      if line:
        inference_data.append(u' '.join(list(line)))

  return inference_data


def _decode_by_function(writer_function, text_raw, decoded_tags, seq_lens):
  assert len(text_raw) == len(decoded_tags)
  assert len(seq_lens) == len(decoded_tags)

  for text_line, tags_line, length in zip(text_raw, decoded_tags, seq_lens):
    text_line = text_line[ : length]
    tags_line = tags_line[ : length]
    newline = u""

    for char, tag in zip(text_line, tags_line):
      char = char.decode("utf-8")
      if tag == TAG_S or tag == TAG_B:
        newline += u' ' + char
      else:
        newline += char

    newline = newline.strip()
    writer_function(newline + u'\n')


def inference(ckpt, input_file, trans_file, hparams, model_creator):
  infer_model = model_helper.create_infer_model(hparams, model_creator)

  infer_sess = tf.Session(graph = infer_model.graph)
  with infer_model.graph.as_default():
    loaded_infer_model = model_helper.load_model(infer_model.model,
        ckpt, infer_sess, "infer", init = True)

  # Read data
  infer_data = load_data(input_file)
  infer_sess.run(
        infer_model.iterator.initializer,
        feed_dict = {
            infer_model.txt_placeholder: infer_data,
            infer_model.batch_size_placeholder: hparams.infer_batch_size
        })

  with codecs.getwriter("utf-8")(
        tf.gfile.GFile(trans_file, mode="wb")) as trans_f:
    while True:
      try:
        text_raw, decoded_tags, seq_lens = loaded_infer_model.infer(infer_sess)
      except tf.errors.OutOfRangeError:
        # finish the evaluation
        break
      _decode_by_function(lambda x : trans_f.write(x), text_raw, decoded_tags, seq_lens)