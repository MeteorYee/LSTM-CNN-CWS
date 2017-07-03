# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 06/27/2017
#
# Description: generate training for char pos
#
# Last Modified at: 06/27/2017, by: Synrey Yee


import SentHandler as snhd
import cPickle

MAX_LEN = 80
Test_Size = 8000
step = 50

corpusAll = "/home/meteor/data/people2014All.txt"
# corpusAll = "/home/meteor/data/ctb7_update_utf8.txt"
vecpath = "char_vec.txt"
pos_tag_path = "pos_vocab.txt"

posip = open(pos_tag_path, 'r')
pos_dict = {}
pos_dict["UNK"] = 0
index = 1

for line in posip:
	pair = line.strip().split()
	pos_dict[pair[0]] = index
	index += 1

# print pos_dict
posip.close()

train_file = "Corpora/cpos_train.txt"
test_file = "Corpora/cpos_test.txt"

test_file_raw = "Corpora/cpos_test_raw.txt"
test_file_gold = "Corpora/cpos_test_gold.txt"

vob = open(vecpath, 'r')
lines = vob.readlines()
first_line = lines[0].strip()
words_num = int(first_line.split()[0])

vob_dict = {}
for i in xrange(words_num):
	line = lines[i + 1].strip()
	word = line.split()[0].decode("utf-8")
	vob_dict[word] = i

vob.close()

inp = open(corpusAll, 'r')
bad_lines = 0
cnt = 0

trop_ = open(train_file, 'w')
ttop_ = open(test_file, 'w')
ttrop_ = open(test_file_raw, 'w')
ttgop_ = open(test_file_gold, 'w')
with trop_ as trop, ttop_ as ttop, ttrop_ as ttrop, ttgop_ as ttgop:
	for ind, line in enumerate(inp):
		line_pieces = []
		NE_free_line = snhd.NE_Removing(line)
		flag = snhd.SliceSentence(NE_free_line, line_pieces, [], tag = True, max_len = MAX_LEN)

		if not flag:
			bad_lines += 1
			continue

		analyzed_pieces, flag = snhd.POS_Analyze(line_pieces, vob_dict, pos_dict, max_len = MAX_LEN)

		if not flag:
			bad_lines += 1
			continue

		if (ind + 1) % step == 0 and cnt < Test_Size:
			for piece_raw, piece in zip(line_pieces, analyzed_pieces):
				ttrop.write(snhd.CleanSentence(piece_raw, set([])))
				ttgop.write(piece_raw)
				ttop.write(piece)
			cnt += 1
		else:
			for piece in analyzed_pieces:
				trop.write(piece)

	print "Generating finished, gave up %d bad lines" % bad_lines

inp.close()