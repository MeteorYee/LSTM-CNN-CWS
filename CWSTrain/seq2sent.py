# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 07/08/2017
#
# Description: transform the result sequences to sentences
#
# Last Modified at: 07/08/2017, by: Synrey Yee

import sys

if __name__ == '__main__':
	seq_file = sys.argv[1]
	raw_file = sys.argv[2]
	sent_file = sys.argv[3]

	seq_inp = open(seq_file, 'r')
	sequences = seq_inp.readlines()
	rinp = open(raw_file, 'r')

	with open(sent_file, 'w') as opt:
		for ind, line in enumerate(rinp):
			ustr = line.strip().decode("utf-8")
			seq = sequences[ind].strip().split()
			newline = u""
			for word, label in zip(ustr, seq):
				if label == '0' or label == '1':
					newline += u' ' + word
				else:
					newline += word

			newline = newline.strip().encode("utf-8")
			opt.write(newline + '\n')

	seq_inp.close()
	rinp.close()
