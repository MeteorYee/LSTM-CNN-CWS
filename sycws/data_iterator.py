# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 03/24/2018
#
# Description: The iterator of training and inference data
#
# Last Modified at: 04/03/2018, by: Synrey Yee

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

from collections import namedtuple

import tensorflow as tf

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


class BatchedInput(
    namedtuple("BatchedInput",
               ("initializer", "text", "label",
                "text_raw", "sequence_length"))):
  pass


def get_infer_iterator(src_dataset,
                       vocab_table,
                       index_table,
                       batch_size,
                       max_len = None):

  src_dataset = src_dataset.map(lambda src : tf.string_split([src]).values)

  if max_len:
    src_dataset = src_dataset.map(lambda src : src[:max_len])
  # Convert the word strings to ids
  src_dataset = src_dataset.map(
      lambda src : (src, tf.cast(vocab_table.lookup(src), tf.int32)))
  # Add in the word counts.
  # Add the fake labels, refer to model_helper.py line 160.
  src_dataset = src_dataset.map(lambda src_raw, src : (src_raw, src,
    tf.cast(index_table.lookup(src_raw), tf.int32), tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # src_raw
            tf.TensorShape([None]),  # src_ids
            tf.TensorShape([None]),  # fake label ids
            tf.TensorShape([])),  # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.

        # padding_values = 0, default value
        )

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_raw, src_ids, lb_ids, src_seq_len) = batched_iter.get_next()
  return BatchedInput(
      initializer = batched_iter.initializer,
      text = src_ids,
      label = lb_ids,
      text_raw = src_raw,
      sequence_length = src_seq_len)


def get_iterator(txt_dataset,
                 lb_dataset,
                 vocab_table,
                 index_table,
                 batch_size,
                 num_buckets,
                 max_len = None,
                 output_buffer_size = None,
                 num_parallel_calls = 4):

  if not output_buffer_size:
    output_buffer_size = batch_size * 1000
  txt_lb_dataset = tf.data.Dataset.zip((txt_dataset, lb_dataset))
  txt_lb_dataset = txt_lb_dataset.shuffle(output_buffer_size)

  txt_lb_dataset = txt_lb_dataset.map(
      lambda txt, lb : (
          tf.string_split([txt]).values, tf.string_split([lb]).values),
      num_parallel_calls = num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  txt_lb_dataset = txt_lb_dataset.filter(
      lambda txt, lb : tf.logical_and(tf.size(txt) > 0, tf.size(lb) > 0))

  if max_len:
    txt_lb_dataset = txt_lb_dataset.map(
        lambda txt, lb : (txt[ : max_len], lb[ : max_len]),
        num_parallel_calls = num_parallel_calls).prefetch(output_buffer_size)
  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.

  txt_lb_dataset = txt_lb_dataset.map(
      lambda txt, lb : (
          tf.cast(vocab_table.lookup(txt), tf.int32),
          tf.cast(index_table.lookup(lb), tf.int32)),
      num_parallel_calls = num_parallel_calls).prefetch(output_buffer_size)
  
  # Add in sequence lengths.
  txt_lb_dataset = txt_lb_dataset.map(
      lambda txt, lb : (
          txt, lb, tf.size(txt)),
      num_parallel_calls = num_parallel_calls).prefetch(output_buffer_size)

  # Bucket by sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first two entries are the text and label line rows;
        # these have unknown-length vectors.  The last entry is
        # the row sizes; these are scalars.
        padded_shapes = (
            tf.TensorShape([None]),  # txt
            tf.TensorShape([None]),  # lb
            tf.TensorShape([])),  # length
        # Pad the text and label sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.

        padding_values = (-1, -1, 0) # distinguish it from UNK_id,
        # 0 for length--unused though
        )

  if num_buckets > 1:

    def key_func(unused_1, unused_2, seq_len):
      # Calculate bucket_width by maximum text sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if max_len:
        bucket_width = (max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 25

      # Bucket sentence pairs by the length of their text sentence and label
      # sentence.
      bucket_id = seq_len // bucket_width
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = txt_lb_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func = key_func, reduce_func = reduce_func, window_size = batch_size))

    # One batch could have multiple windows, although there is just one window
    # in a batch.

  else:
    batched_dataset = batching_func(txt_lb_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (txt_ids, lb_ids, seq_len) = batched_iter.get_next()
  return BatchedInput(
      initializer = batched_iter.initializer,
      text = txt_ids,
      label = lb_ids,
      text_raw = None,
      sequence_length = seq_len)