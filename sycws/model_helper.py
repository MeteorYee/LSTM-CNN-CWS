# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 03/24/2018
#
# Description: Help functions for the implementation of the models
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

from __future__ import absolute_import
from __future__ import print_function

from collections import namedtuple
from tensorflow.python.ops import lookup_ops

from . import data_iterator

import tensorflow as tf
import numpy as np
import time
import codecs

__all__ = [
    "create_train_model", "create_eval_model",
    "create_infer_model", "create_cnn_layer"
    "create_or_load_model", "load_model",
    "create_pretrained_emb_from_txt"
]


UNK_ID = 0


# subclass to purify the __dict__
class TrainModel(
    namedtuple("TrainModel", ("graph", "model",
        "iterator"))):
  pass


def create_train_model(hparams, model_creator):
  txt_file = "%s.%s" % (hparams.train_prefix, "txt")
  lb_file = "%s.%s" % (hparams.train_prefix, "lb")
  vocab_file = hparams.vocab_file
  index_file = hparams.index_file

  graph = tf.Graph()

  with graph.as_default(), tf.container("train"):
    vocab_table = lookup_ops.index_table_from_file(
      vocab_file, default_value = UNK_ID)
    # for the labels
    index_table = lookup_ops.index_table_from_file(
      index_file, default_value = 0)

    txt_dataset = tf.data.TextLineDataset(txt_file)
    lb_dataset = tf.data.TextLineDataset(lb_file)

    iterator = data_iterator.get_iterator(
        txt_dataset,
        lb_dataset,
        vocab_table,
        index_table,
        batch_size = hparams.batch_size,
        num_buckets = hparams.num_buckets,
        max_len = hparams.max_len)

    model = model_creator(
        hparams,
        iterator = iterator,
        mode = tf.contrib.learn.ModeKeys.TRAIN,
        vocab_table = vocab_table)

  return TrainModel(
      graph = graph,
      model = model,
      iterator = iterator)


class EvalModel(
    namedtuple("EvalModel",
               ("graph", "model", "txt_file_placeholder",
                "lb_file_placeholder", "iterator"))):
  pass


def create_eval_model(hparams, model_creator):
  vocab_file = hparams.vocab_file
  index_file = hparams.index_file
  graph = tf.Graph()

  with graph.as_default(), tf.container("eval"):
    vocab_table = lookup_ops.index_table_from_file(
      vocab_file, default_value = UNK_ID)
    # for the labels
    index_table = lookup_ops.index_table_from_file(
      index_file, default_value = 0)

    # the file's name
    txt_file_placeholder = tf.placeholder(shape = (), dtype = tf.string)
    lb_file_placeholder = tf.placeholder(shape = (), dtype = tf.string)
    txt_dataset = tf.data.TextLineDataset(txt_file_placeholder)
    lb_dataset = tf.data.TextLineDataset(lb_file_placeholder)

    iterator = data_iterator.get_iterator(
        txt_dataset,
        lb_dataset,
        vocab_table,
        index_table,
        batch_size = hparams.batch_size,
        num_buckets = hparams.num_buckets,
        max_len = hparams.max_len)

    model = model_creator(
        hparams,
        iterator = iterator,
        mode = tf.contrib.learn.ModeKeys.EVAL,
        vocab_table = vocab_table)

  return EvalModel(
      graph = graph,
      model = model,
      txt_file_placeholder = txt_file_placeholder,
      lb_file_placeholder = lb_file_placeholder,
      iterator = iterator)


class InferModel(
    namedtuple("InferModel",
               ("graph", "model", "txt_placeholder",
                "batch_size_placeholder", "iterator"))):
  pass


def create_infer_model(hparams, model_creator):
  """Create inference model."""
  graph = tf.Graph()
  vocab_file = hparams.vocab_file

  with graph.as_default(), tf.container("infer"):
    vocab_table = lookup_ops.index_table_from_file(
      vocab_file, default_value = UNK_ID)
    # for the labels
    '''
    Although this is nonsense for the inference procedure, this is to ensure
    the labels are not None when building the model graph.
    (refer to model.BasicModel._decode_layer)
    '''
    mapping_strings = tf.constant(['0'])
    index_table = tf.contrib.lookup.index_table_from_tensor(
    mapping = mapping_strings, default_value = 0)

    txt_placeholder = tf.placeholder(shape=[None], dtype = tf.string)
    batch_size_placeholder = tf.placeholder(shape = [], dtype = tf.int64)

    txt_dataset = tf.data.Dataset.from_tensor_slices(
        txt_placeholder)
    iterator = data_iterator.get_infer_iterator(
        txt_dataset,
        vocab_table,
        index_table,
        batch_size = batch_size_placeholder)

    model = model_creator(
        hparams,
        iterator = iterator,
        mode = tf.contrib.learn.ModeKeys.INFER,
        vocab_table = vocab_table)

  return InferModel(
      graph = graph,
      model = model,
      txt_placeholder = txt_placeholder,
      batch_size_placeholder = batch_size_placeholder,
      iterator = iterator)


def _load_vocab(vocab_file):
  vocab = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    vocab_size = 0
    for word in f:
      vocab_size += 1
      vocab.append(word.strip())
  return vocab, vocab_size


def _load_embed_txt(embed_file):
  """Load embed_file into a python dictionary.

  Note: the embed_file should be a Glove formated txt file. Assuming
  embed_size=5, for example:

  the -0.071549 0.093459 0.023738 -0.090339 0.056123
  to 0.57346 0.5417 -0.23477 -0.3624 0.4037
  and 0.20327 0.47348 0.050877 0.002103 0.060547

  Note: The first line stores the information of the # of embeddings and
  the size of an embedding.

  Args:
    embed_file: file path to the embedding file.
  Returns:
    a dictionary that maps word to vector, and the size of embedding dimensions.
  """
  emb_dict = dict()
  with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
    emb_num, emb_size = f.readline().strip().split()
    emb_num = int(emb_num)
    emb_size = int(emb_size)
    for line in f:
      tokens = line.strip().split(" ")
      word = tokens[0]
      vec = list(map(float, tokens[1:]))
      emb_dict[word] = vec
      assert emb_size == len(vec), "All embedding size should be same."
  return emb_dict, emb_size


def create_pretrained_emb_from_txt(vocab_file, embed_file, dtype = tf.float32):
  """Load pretrain embeding from embed_file, and return an embedding matrix.

  Args:
    embed_file: Path to a Glove formated embedding txt file.
    Note: we only need the embeddings whose corresponding words are in the
    vocab_file.
  """
  vocab, _ = _load_vocab(vocab_file)

  print('# Using pretrained embedding: %s.' % embed_file)
  emb_dict, emb_size = _load_embed_txt(embed_file)

  emb_mat = np.array(
      [emb_dict[token] for token in vocab], dtype = dtype.as_numpy_dtype())

  # The commented codes below are used for creating untrainable embeddings, which means
  # the value of each embedding is settled.
  # num_trainable_tokens = 1 # the unk token is trainable
  # emb_mat = tf.constant(emb_mat)
  # emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
  # with tf.variable_scope("pretrain_embeddings", dtype = dtype) as scope:
  #   emb_mat_var = tf.get_variable(
  #       "emb_mat_var", [num_trainable_tokens, emb_size])
  # return tf.concat([emb_mat_var, emb_mat_const], 0)

  # Whereas we use the pretrained embedding values as initial values,
  # so the embeddings can be trainable and their values can be changed.
  return tf.Variable(emb_mat, name = "char_embedding")


def _char_convolution(inputs, cfilter):
  conv1 = tf.nn.conv2d(inputs, cfilter, [1, 1, 1, 1],
                         padding = 'VALID')
  # inputs.shape = [batch_size, 1, 3, 2*num_units]
  # namely, in_height = 1, in_width = 3, in_channels = 2*num_units
  # filter.shape = [1, 2, 2*num_units, num_tags], strides = [1, 1, 1, 1]
  # conv1.shape = [batch_size, 1, 2, num_tags]
  conv1 = tf.nn.relu(conv1)
  pool1 = tf.nn.max_pool(conv1,
      ksize = [1, 1, 2, 1],
      strides = [1, 1, 1, 1],
      padding = 'VALID')

  # pool1.shape = [batch_size, 1, 1, num_tags]
  pool1 = tf.squeeze(pool1, [1, 2])
  # pool1.shape = [batch_size, num_tags]
  return pool1


def create_cnn_layer(inputs, time_major, batch_size, num_units, cfilter):
  if not time_major:
    # trnaspose
    inputs = tf.trnaspose(inputs, [1, 0, 2])

  inputs = tf.expand_dims(inputs, 2)
  # [max_time, batch_size, 1, 2*num_units]

  left = inputs[1 : ]
  right = inputs[ : -1]
  left = tf.pad(left, [[1, 0], [0, 0], [0, 0], [0, 0]], "CONSTANT")
  right = tf.pad(right, [[0, 1], [0, 0], [0, 0], [0, 0]], "CONSTANT")

  char_blocks = tf.concat([left, inputs, right], 3)
  # [max_time, batch_size, 1, 3*2*num_units]
  char_blocks = tf.reshape(char_blocks, [-1, batch_size, 3, (2 * num_units)])
  char_blocks = tf.expand_dims(char_blocks, 2)
  # [max_time, batch_size, 1, 3, 2*num_units]

  do_char_conv = lambda x : _char_convolution(x, cfilter)
  cnn_outputs = tf.map_fn(do_char_conv, char_blocks)
  # [max_time, batch_size, num_tags]

  return tf.transpose(cnn_outputs, [1, 0, 2])


def load_model(model, ckpt, session, name, init):
  start_time = time.time()
  model.saver.restore(session, ckpt)
  if init:
    session.run(tf.tables_initializer())
  print(
      "  loaded %s model parameters from %s, time %.2fs" %
      (name, ckpt, time.time() - start_time))
  return model


def create_or_load_model(model, model_dir, session, name, init):
  """Create segmentation model and initialize or load parameters in session."""
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    model = load_model(model, latest_ckpt, session, name, init)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print("  created %s model with fresh parameters, time %.2fs" %
                    (name, time.time() - start_time))

  global_step = model.global_step.eval(session = session)
  return model, global_step