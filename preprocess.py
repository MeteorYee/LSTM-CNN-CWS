# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 04/24/2017
#
# Description: pre-processing
#
# Last Modified at: 05/28/2017, by: Synrey Yee

import SentHandler as snhd
import cPickle

# this path depends on your choice
corpusAll = "/home/meteor/data/people2014All.txt"

flag = True
# if want to preprocess NER or POS tagging, set flag to 'False'
# flag = False

# if all the corpus is not in one file
# snhd.All2oneFile(rootDir, corpusAll)
# where 'rootDir' is corpus' root name

# used to collect tags
tags = set([])

# its name is up to you
result_file = "pre_words_for_w2v.txt"

with open(result_file, 'w') as opt:
	inp = open(corpusAll, 'r')
	for line in inp:
		NE_free_line = snhd.NE_Removing(line)
		newline = snhd.CleanSentence(NE_free_line, tags, interval = u' ', breakword = flag)
		opt.write(newline)

	inp.close()

print "got %d tags" % len(tags)
with open("Models/pos_tags.cpk", 'w') as opt:
	cPickle.dump(tags, opt)
