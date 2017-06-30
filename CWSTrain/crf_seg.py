# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 05/14/2017
#
# Description: segmentation script
#
# Last Modified at: 05/28/2017, by: Synrey Yee

import tensorflow as tf
import numpy as np
import bmes_train_lstm as btl


tf.app.flags.DEFINE_string('test_path', "Corpora/ppl_test_raw.txt",
                           'test data, waiting to be segmented')
tf.app.flags.DEFINE_string('result_path', "Results/ppl_bmes_result.txt",
                           'results data dir')
MAX_LEN = btl.FLAGS.max_sentence_len

# make the raw data acceptable for the model
def TransRawData(test_data, vob_dict):
	inp = open(test_data, 'r')
	X = []
	Y = []
	for line in inp:
		ustr = line.decode("utf-8").strip()
		lX = []
		lY = [0 for i in xrange(MAX_LEN)]
		for char in ustr:
			if vob_dict.has_key(char):
				lX.append(vob_dict[char])
			else:
				lX.append(vob_dict[u"<UNK>"])

		for _ in xrange(len(ustr), MAX_LEN):
			lX.append(0)

		X.append(lX)
		Y.append(lY)

	inp.close()
	return np.array(X), np.array(Y)

def segment_seq(test_data, word2vec_path, vob_dict):
	X, hidden_W, hidden_b = btl.initialization(word2vec_path)
	Y = tf.placeholder(tf.int32, shape = [None, MAX_LEN])

	tX, tY = TransRawData(test_data, vob_dict)

	P, sequence_length = btl.inference(X, hidden_W, hidden_b)

	log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
		P, Y, sequence_length)

	test_unary_score, test_sequence_length = btl.inference(X,
			hidden_W, hidden_b, reuse = True, trainMode = False)

	results = []
	sv = tf.train.Supervisor(logdir = btl.FLAGS.log_dir)
	with sv.managed_session(master = '') as sess:
		# we need this to do evaluation
		_, trainsMatrix = sess.run([log_likelihood, transition_params],
								feed_dict = {X : tX, Y : tY})

		unary_scores, seq_lens = sess.run([test_unary_score, test_sequence_length], feed_dict = {X : tX})
		for unary_score, slen in zip(unary_scores, seq_lens):
			viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                unary_score[ : slen], trainsMatrix)
			results.append(viterbi_sequence)

	return results

def main(unused_argv):
	vob = open(btl.FLAGS.word2vec_path, 'r')
	lines = vob.readlines()
	first_line = lines[0].strip()
	words_num = int(first_line.split()[0])

	vob_dict = {}
	for i in xrange(words_num):
		line = lines[i + 1].strip()
		word = line.split()[0].decode("utf-8")
		vob_dict[word] = i

	vob.close()

	sequences = segment_seq(btl.FLAGS.test_path, btl.FLAGS.word2vec_path, vob_dict)

	rinp = open(btl.FLAGS.test_path, 'r')
	with open(btl.FLAGS.result_path, 'w') as opt:
		for ind, line in enumerate(rinp):
			ustr = line.strip().decode("utf-8")
			seq = sequences[ind]
			newline = u""
			for word, label in zip(ustr, seq):
				if label == 0 or label == 1:
					newline += u' ' + word
				else:
					newline += word

			newline = newline.strip().encode("utf-8")
			opt.write(newline + '\n')

	rinp.close()

if __name__ == '__main__':
	tf.app.run()