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
import lstm_cnn_train as lct


tf.app.flags.DEFINE_string('test_path', "Corpora_cityu/test_raw.txt",
                           'test data, waiting to be segmented')
tf.app.flags.DEFINE_string('result_path', "Corpora_cityu/cnn_result_12k.txt",
                           'results data dir')
MAX_LEN = lct.FLAGS.max_sentence_len

# make the raw data acceptable for the model
def TransRawData(test_data, vob_dict):
	inp = open(test_data, 'r')
	X = []
	for line in inp:
		ustr = line.decode("utf-8").strip()
		lX = []
		for char in ustr:
			if vob_dict.has_key(char):
				lX.append(vob_dict[char])
			else:
				lX.append(vob_dict[u"<UNK>"])

		for _ in xrange(len(ustr), MAX_LEN):
			lX.append(0)

		X.append(lX)

	inp.close()
	return np.array(X)

def segment_seq(test_data, word2vec_path, vob_dict):
	X, hidden_W, hidden_b = lct.initialization(word2vec_path)

	tX = TransRawData(test_data, vob_dict)

	P, _ = lct.inference(X, hidden_W, hidden_b)

	test_four_scores, test_sequence_length = lct.inference(X,
		hidden_W, hidden_b, reuse = True, trainMode = False)
	test_unary_score = tf.argmax(test_four_scores, 2)

	results = []
	sv = tf.train.Supervisor(logdir = lct.FLAGS.log_dir)
	batchSize = lct.FLAGS.batch_size
	with sv.managed_session(master = '') as sess:
		'''
		We need this to do evaluation, which means wanking up our parameters
		using one batch.
		'''
		sess.run(P, feed_dict = {X : tX[0 : batchSize]})
		'''
		Note Here!
		We can't just set feed_dict = {X : tX}, because when tX is too
		big, tensorflow cannot handle it normally. For example, on my
		computer, I wrote it in this way, and my system halted... 

		Hence, the following codes are doing in a same way.
		'''
		totalLen = tX.shape[0]
		numBatch = int((totalLen - 1) / batchSize) + 1

		for i in xrange(numBatch):
			endOff = (i + 1) * batchSize
			if endOff > totalLen:
				endOff = totalLen

			feed_dict = {X : tX[i * batchSize : endOff]}
			unary_scores, seq_lens = sess.run([test_unary_score, test_sequence_length], feed_dict)

			for unary_score, slen in zip(unary_scores, seq_lens):
				best_sequence = unary_score[ : slen]
				results.append(best_sequence)

	return results

def main(unused_argv):
	vob = open(lct.FLAGS.word2vec_path, 'r')
	lines = vob.readlines()
	first_line = lines[0].strip()
	words_num = int(first_line.split()[0])

	vob_dict = {}
	for i in xrange(words_num):
		line = lines[i + 1].strip()
		word = line.split()[0].decode("utf-8")
		vob_dict[word] = i

	vob.close()

	sequences = segment_seq(lct.FLAGS.test_path, lct.FLAGS.word2vec_path, vob_dict)

	rinp = open(lct.FLAGS.test_path, 'r')
	with open(lct.FLAGS.result_path, 'w') as opt:
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