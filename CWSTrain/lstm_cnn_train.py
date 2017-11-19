# -*- coding: utf-8 -*-
# @Author: Koth Chen
# @Date:   2016-07-26 13:48:32
# @Last Modified by:   Synrey Yee
# @Last Modified time: 2017-07-06

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import os
import pdb

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path', "Corpora/train.txt",
                           'Training data dir')
tf.app.flags.DEFINE_string('test_data_path', "Corpora/test.txt",
                           'Test data dir')
tf.app.flags.DEFINE_string('log_dir', "Logs/seg_cnn_logs", 'The log dir')
tf.app.flags.DEFINE_string("word2vec_path", "char_vec.txt",
                           "the word2vec data path")

tf.app.flags.DEFINE_integer("max_sentence_len", 80,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("embedding_size", 50, "embedding size")
tf.app.flags.DEFINE_integer("num_tags", 4, "BMES")
tf.app.flags.DEFINE_integer("mrank", 1, "Markov rank")
tf.app.flags.DEFINE_integer("num_hidden", 100, "hidden unit number")
tf.app.flags.DEFINE_integer("batch_size", 100, "num example per mini batch")
tf.app.flags.DEFINE_integer("train_steps", 50000, "trainning steps")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")

# the word2vec words
WORDS = None
# convolution's filter
cfilter = None

def initialization(c2vPath):
    c2v = load_w2v(c2vPath, FLAGS.embedding_size)

    global WORDS
    WORDS = tf.Variable(c2v, name = "words")

    inp = tf.placeholder(tf.int32,
                              shape = [None, FLAGS.max_sentence_len],
                              name = "input_placeholder")

    with tf.variable_scope('Softmax') as scope:
        hidden_W = tf.get_variable(
            shape = [FLAGS.num_hidden * 2, FLAGS.num_tags],
            initializer = tf.truncated_normal_initializer(stddev = 0.01),
            name = "weights",
            regularizer = tf.contrib.layers.l2_regularizer(0.001))

        hidden_b = tf.Variable(tf.zeros([FLAGS.num_tags], name = "bias"))

    global cfilter
    with tf.variable_scope('CNN_Layer') as scope:
        cfilter = tf.get_variable(
            "cfilter",
            shape = [FLAGS.mrank + 1, 2 * FLAGS.num_hidden, 1, 2 * FLAGS.num_hidden],
            regularizer = tf.contrib.layers.l2_regularizer(0.0001),
            initializer = tf.truncated_normal_initializer(stddev = 0.01),
            dtype = tf.float32)

    return inp, hidden_W, hidden_b

def GetLength(data):
    used = tf.sign(tf.abs(data))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def char_convolution(vecs):
    global cfilter

    conv1 = tf.nn.conv2d(vecs, cfilter, [1, 1, 2 * FLAGS.num_hidden, 1],
                         padding='VALID')
    # vecs.shape = [batch_size, 3, 200, 1] (for 1-rank Markov assumption)
    # filter.shape = [2, 200, 1, 200], strides = [1, 1, 200, 1]
    # conv1.shape = [batch_size, 2, 1, 200]
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1,
        ksize = [1, FLAGS.mrank + 1, 1, 1],
        strides = [1, FLAGS.mrank + 1, 1, 1],
        padding = 'SAME')

    # ksize.shape = [1, 2, 1, 1], strides.shape = [1, 2, 1, 1]
    # pool1.shape = [batch_size, 1, 1, 200]
    pool1 = tf.squeeze(pool1, [1, 2])
    # pool1.shape = [batch_size, 200]
    return pool1

def inference(X, weights, bias, reuse = None, trainMode = True):
    word_vectors = tf.nn.embedding_lookup(WORDS, X)
    # [batch_size, 80, 50]

    length = GetLength(X)
    length_64 = tf.cast(length, tf.int64)
    reuse = None if trainMode else True

    with tf.variable_scope("rnn_fwbw", reuse = reuse) as scope:
        forward_output, _ = tf.nn.dynamic_rnn(
            tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, reuse = reuse),
            word_vectors,
            dtype = tf.float32,
            sequence_length = length,
            scope = "RNN_forward")
        backward_output_, _ = tf.nn.dynamic_rnn(
            tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, reuse = reuse),
            inputs = tf.reverse_sequence(word_vectors,
                                       length_64,
                                       seq_dim = 1),
            dtype = tf.float32,
            sequence_length = length,
            scope = "RNN_backword")

    backward_output = tf.reverse_sequence(backward_output_,
                                          length_64,
                                          seq_dim = 1)
    
    output = tf.concat([forward_output, backward_output], 2)
    # [batch_size, 80, 200]

    output = tf.expand_dims(output, -1)
    # [batch_size, 80, 200, 1]
    output = tf.transpose(output, perm = [1, 0, 3, 2])
    # [80, batch_size, 1, 200]

    char_blocks = output
    for i in range(FLAGS.mrank):
        ileft = output[(i + 1) : ]
        iright = output[ : FLAGS.max_sentence_len - (i + 1)]

        ileft = tf.pad(ileft, [[(i + 1), 0], [0, 0], [0, 0], [0, 0]], "CONSTANT")
        iright = tf.pad(iright, [[0, (i + 1)], [0, 0], [0, 0], [0, 0]], "CONSTANT")
        
        char_blocks = tf.concat([ileft, char_blocks, iright], 3)
    # char_blocks.shape = [80, batch_size, 1, 200 * (2*mrank + 1)]

    char_blocks = tf.reshape(char_blocks, [FLAGS.max_sentence_len, -1,
                                            2 * FLAGS.mrank + 1,
                                            2 * FLAGS.num_hidden])
    char_blocks = tf.expand_dims(char_blocks, -1)
    # [80, batch_size, 2 * mrank + 1, 200, 1]
    # Namely, [80, batch_size, 3, 200, 1] for 1-rank Markov assumption

    # do conv
    do_char_conv = lambda x: char_convolution(x)
    abstract_chars = tf.map_fn(do_char_conv, char_blocks)
    # [80, batch_size, 200]

    abstract_chars = tf.transpose(abstract_chars, perm = [1, 0, 2])
    abstract_chars = tf.reshape(abstract_chars, [-1, FLAGS.num_hidden * 2])
    if trainMode:
        abstract_chars = tf.nn.dropout(abstract_chars, 0.5)

    matricized_unary_scores = tf.matmul(abstract_chars, weights) + bias

    unary_scores = tf.reshape(matricized_unary_scores,
            [-1, FLAGS.max_sentence_len, FLAGS.num_tags])

    # [batch_size, 80, 4]
    return unary_scores, length

def load_w2v(path, expectDim):
    fp = open(path, "r")
    print("load data from:", path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    assert (dim == expectDim)
    ws = []
    mv = [0 for i in range(dim)]
    second = -1
    for t in range(total):
        if ss[0] == '<UNK>':
            second = t
        line = fp.readline().strip()
        ss = line.split(" ")
        assert (len(ss) == (dim + 1))
        vals = []
        for i in range(1, dim + 1):
            fv = float(ss[i])
            mv[i - 1] += fv
            vals.append(fv)
        ws.append(vals)
    for i in range(dim):
        mv[i] = mv[i] / total
    assert (second != -1)
    # append one more token , maybe useless
    ws.append(mv)
    if second != 1:
        t = ws[1]
        ws[1] = ws[second]
        ws[second] = t
    fp.close()
    return np.asarray(ws, dtype=np.float32)

def do_load_data(path):
    x = []
    y = []
    fp = open(path, "r")
    for line in fp.readlines():
        line = line.rstrip()
        if not line:
            continue
        ss = line.split(" ")
        assert (len(ss) == (FLAGS.max_sentence_len * 2))
        lx = []
        ly = []
        for i in range(FLAGS.max_sentence_len):
            lx.append(int(ss[i]))
            ly.append(int(ss[i + FLAGS.max_sentence_len]))
        x.append(lx)
        y.append(ly)
    fp.close()
    return np.array(x), np.array(y)

def read_csv(batch_size, file_name):
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(filename_queue)
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(
        value,
        field_delim=' ',
        record_defaults=[[0] for i in range(FLAGS.max_sentence_len * 2)])

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)

def test_evaluate(sess, unary_score, test_sequence_length, inp, tX, tY):
    totalEqual = 0
    batchSize = FLAGS.batch_size
    totalLen = tX.shape[0]
    numBatch = int((tX.shape[0] - 1) / batchSize) + 1
    correct_labels = 0
    total_labels = 0

    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen

        y = tY[i * batchSize:endOff]
        feed_dict = {inp: tX[i * batchSize:endOff]}
        unary_score_val, test_sequence_length_val = sess.run(
            [unary_score, test_sequence_length], feed_dict)

        for tf_unary_scores_, y_, sequence_length_ in zip(
                unary_score_val, y, test_sequence_length_val):

            best_sequence = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]

            # Evaluate word-level accuracy.
            correct_labels += np.sum(np.equal(best_sequence, y_))
            total_labels += sequence_length_

    cl = np.float64(correct_labels)
    tl = np.float64(total_labels)
    accuracy = 100.0 * cl / tl

    print("Accuracy: %.3f%%" % accuracy)
    return accuracy

def inputs(path):
    whole = read_csv(FLAGS.batch_size, path)
    features = tf.transpose(tf.stack(whole[0:FLAGS.max_sentence_len]))
    label = tf.transpose(tf.stack(whole[FLAGS.max_sentence_len:]))
    return features, label

def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)


def main(unused_argv):
    trainDataPath = FLAGS.train_data_path

    graph = tf.Graph()
    with graph.as_default():
        inp, hidden_W, hidden_b = initialization(FLAGS.word2vec_path)

        print("train data path:", trainDataPath)

        X, Y = inputs(trainDataPath)
        tX, tY = do_load_data(FLAGS.test_data_path)

        P, sequence_length = inference(X, hidden_W, hidden_b)

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = P, labels = Y)
        cross_entropy = tf.reduce_mean(xentropy)

        train_op = train(cross_entropy)
        test_four_scores, test_sequence_length = inference(inp,
           hidden_W, hidden_b, reuse = True, trainMode = False)
        test_unary_score = tf.argmax(test_four_scores, 2)

        sv = tf.train.Supervisor(graph = graph, logdir = FLAGS.log_dir)
        with sv.managed_session(master = '') as sess:
            dev = np.float64(0.0001)
            last_acc = np.float64(0)
            # actual training loop
            training_steps = FLAGS.train_steps
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    sess.run(train_op)
                    # for debugging and learning purposes, see how the loss gets decremented thru training steps
                    if (step + 1) % 100 == 0:
                        print("[%d] loss: [%r]" %
                              (step + 1, sess.run(cross_entropy)))
                    if (step + 1) % 1000 == 0:
                        test_evaluate(sess, test_unary_score,
                            test_sequence_length, inp, tX, tY)

                except KeyboardInterrupt as e:
                    sv.saver.save(sess,
                                  FLAGS.log_dir + '/model',
                                  global_step = (step + 1))
                    raise e

            sv.saver.save(sess, FLAGS.log_dir + '/finnal-model')


if __name__ == '__main__':
    tf.app.run()
