# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 03/24/2018
#
# Description: Segmentation models
#
# Last Modified at: 04/08/2018, by: Synrey Yee

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

from . import model_helper
from . import data_iterator

import tensorflow as tf


class BasicModel(object):
  """Bi-LSTM + single layer + CRF"""

  def __init__(self, hparams, mode, iterator, vocab_table):
    assert isinstance(iterator, data_iterator.BatchedInput)

    self.iterator = iterator
    self.mode = mode
    self.vocab_table = vocab_table
    self.vocab_size = hparams.vocab_size
    self.time_major = hparams.time_major

    # Initializer
    self.initializer = tf.truncated_normal_initializer

    # Embeddings
    self.init_embeddings(hparams)
    # Because usually the last batch may be less than the batch
    # size we set, we should get batch_size dynamically.
    self.batch_size = tf.size(self.iterator.sequence_length)

    ## Train graph
    loss = self.build_graph(hparams)

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = loss

    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.char_count = tf.reduce_sum(self.iterator.sequence_length)
      self.right_count = self._calculate_right()

    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.decode_tags = self._decode()

    self.global_step = tf.Variable(0, trainable = False)
    params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model.
    # Arrage for the embedding vars to appear at the beginning.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.learning_rate = tf.constant(hparams.learning_rate)

      # Optimizer, SGD
      opt = tf.train.AdamOptimizer(self.learning_rate)

      # Gradients
      gradients = tf.gradients(self.train_loss, params)

      clipped_grads, grad_norm = tf.clip_by_global_norm(
          gradients, hparams.max_gradient_norm)

      # self.grad_norm = grad_norm
      self.update = opt.apply_gradients(
          zip(clipped_grads, params), global_step = self.global_step)

    # Saver, hparams.num_keep_ckpts
    self.saver = tf.train.Saver(
        tf.global_variables(), max_to_keep = 1)

    # Print trainable variables
    print("# Trainable variables")
    for param in params:
      print("  %s, %s" % (param.name, str(param.get_shape())))


  def init_embeddings(self, hparams, dtype = tf.float32):
    with tf.variable_scope("embeddings", dtype = dtype) as scope:
      self.char_embedding = tf.get_variable(
          "char_embedding", [self.vocab_size, hparams.num_units], dtype,
          initializer = self.initializer(stddev = hparams.init_std))


  def build_graph(self, hparams):
    print("# creating %s graph ..." % self.mode)
    dtype = tf.float32

    with tf.variable_scope("model_body", dtype = dtype):
      # build Bi-LSTM
      encoder_outputs, encoder_state = self._encode_layer(hparams)

      # middle layer
      middle_outputs = self._middle_layer(encoder_outputs, hparams)
      self.middle_outputs = middle_outputs

      # Decoder layer
      xentropy = self._decode_layer(middle_outputs)

      # Loss
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        loss = tf.reduce_mean(xentropy)
      else:
        loss = None
      return loss


  def _encode_layer(self, hparams, dtype = tf.float32):
    # Bi-LSTM
    iterator = self.iterator

    text = iterator.text
    # [batch_size, txt_ids]
    if self.time_major:
      text = tf.transpose(text)
    # [txt_ids, batch_size]

    with tf.variable_scope("encoder", dtype = dtype) as scope:
      # Look up embedding, emp_inp: [max_time, batch_size, num_units]
      encoder_emb_inp = tf.nn.embedding_lookup(
          self.char_embedding, text)

      encoder_outputs, encoder_state = (
          self._build_bidirectional_rnn(
              inputs = encoder_emb_inp,
              sequence_length = iterator.sequence_length,
              dtype = dtype,
              hparams = hparams))
    
    # Encoder_outpus (time_major): [max_time, batch_size, 2*num_units]
    return encoder_outputs, encoder_state


  def _build_bidirectional_rnn(self, inputs, sequence_length,
      dtype, hparams):

    num_units = hparams.num_units
    mode = self.mode
    dropout = hparams.dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

    fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
    bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units)

    if dropout > 0.0:
      fw_cell = tf.contrib.rnn.DropoutWrapper(
          cell = fw_cell, input_keep_prob = (1.0 - dropout))

      bw_cell = tf.contrib.rnn.DropoutWrapper(
          cell = bw_cell, input_keep_prob = (1.0 - dropout))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype = dtype,
        sequence_length = sequence_length,
        time_major = self.time_major)

    # concatenate the fw and bw outputs
    return tf.concat(bi_outputs, -1), bi_state


  def _single_layer(self, encoder_outputs, hparams, dtype):
    num_units = hparams.num_units
    num_tags = hparams.num_tags

    with tf.variable_scope('middle', dtype = dtype) as scope:
      hidden_W = tf.get_variable(
          shape = [num_units * 2, hparams.num_tags],
          initializer = self.initializer(stddev = hparams.init_std),
          name = "weights",
          regularizer = tf.contrib.layers.l2_regularizer(0.001))

      hidden_b = tf.Variable(tf.zeros([num_tags], name = "bias"))

    encoder_outputs = tf.reshape(encoder_outputs, [-1, (2 * num_units)])
    middle_outputs = tf.add(tf.matmul(encoder_outputs, hidden_W), hidden_b)
    if self.time_major:
      middle_outputs = tf.reshape(middle_outputs,
          [-1, self.batch_size, num_tags])
      middle_outputs = tf.transpose(middle_outputs, [1, 0, 2])
    else:
      middle_outputs = tf.reshape(middle_outputs,
          [self.batch_size, -1, num_tags])
    # [batch_size, max_time, num_tags]
    return middle_outputs


  def _cnn_layer(self, encoder_outputs, hparams, dtype):
    num_units = hparams.num_units
    # CNN
    with tf.variable_scope('middle', dtype = dtype) as scope:
      cfilter = tf.get_variable(
          "cfilter",
          shape = [1, 2, 2 * num_units, hparams.num_tags],
          regularizer = tf.contrib.layers.l2_regularizer(0.0001),
          initializer = self.initializer(stddev = hparams.filter_init_std),
          dtype = tf.float32)

    return model_helper.create_cnn_layer(encoder_outputs, self.time_major,
              self.batch_size, num_units, cfilter)


  def _middle_layer(self, encoder_outputs, hparams, dtype = tf.float32):
    # single layer
    return self._single_layer(encoder_outputs, hparams, dtype)


  def _decode_layer(self, middle_outputs, dtype = tf.float32):
    # CRF
    with tf.variable_scope('decoder', dtype = dtype) as scope:
      log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
          middle_outputs, self.iterator.label, self.iterator.sequence_length)

    self.trans_params = trans_params
    return -log_likelihood


  def _decode(self):
    decode_tags, _ = tf.contrib.crf.crf_decode(self.middle_outputs,
        self.trans_params, self.iterator.sequence_length)
    # [batch_size, max_time]

    return decode_tags


  def _calculate_right(self):
    decode_tags = self._decode()
    sign_tensor = tf.equal(decode_tags, self.iterator.label)
    right_count = tf.reduce_sum(tf.cast(sign_tensor, tf.int32))

    return right_count


  def train(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    return sess.run([self.update,
                     self.train_loss,
                     self.global_step,
                     self.batch_size])


  def eval(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    return sess.run([self.char_count,
                     self.right_count,
                     self.batch_size])


  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    return sess.run([self.iterator.text_raw,
                    self.decode_tags,
                    self.iterator.sequence_length])
    

class CnnCrfModel(BasicModel):
  """Bi-LSTM + CNN + CRF"""

  def _middle_layer(self, encoder_outputs, hparams, dtype = tf.float32):
    # CNN
    return self._cnn_layer(encoder_outputs, hparams, dtype)