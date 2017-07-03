# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 05/07/2017
#
# Description: generate training
#
# Last Modified at: 05/11/2017, by: Synrey Yee


import SentHandler as snhd

MAX_LEN = 80
Test_Size = 8000
step = 50

corpusAll = "/home/meteor/data/people2014All.txt"
vecpath = "char_vec.txt"

train_file = "Corpora/train.txt"
test_file = "Corpora/test.txt"
test_file_raw = "Corpora/test_raw.txt"
test_file_gold = "Corpora/test_gold.txt"

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

		analyzed_pieces, flag = snhd.Analyze(line_pieces, vob_dict, max_len = MAX_LEN)

		if not flag:
			bad_lines += 1
			continue

		if (ind + 1) % step == 0 and cnt < Test_Size:
			for piece_raw, piece in zip(line_pieces, analyzed_pieces):
				ttrop.write(snhd.CleanSentence(piece_raw, set([])))
				ttgop.write(snhd.CleanSentence(piece_raw, set([]), interval = u' '))
				ttop.write(piece)
			cnt += 1
		else:
			for piece in analyzed_pieces:
				trop.write(piece)

	print "Generating finished, gave up %d bad lines" % bad_lines

inp.close()