# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 04/26/2017
#
# Description: merge sentences
#
# Last Modified at: 05/06/2017, by: Synrey Yee

def MergeSentence(filename, cut_count, out_file, blacklist = []):
	ipt = open(filename, 'r')
	# add one tail to prevent overflow of cut_count
	cut_count.append(-1)

	with open(out_file, 'w') as opt:
		i = 0
		cnt = cut_count[0]
		for line in ipt:
			if (i + 1) in blacklist and cnt > 0:
				cnt -= 1
				if cnt == 0:
					opt.write("Cannot handle this sentence!\n")
					i += 1
					cnt = cut_count[i]
				continue

			if cnt == 1:
				opt.write(line)
				i += 1
				cnt = cut_count[i]
			else:
				line = line.strip()
				opt.write(line + ' ')
				cnt -= 1

	ipt.close()