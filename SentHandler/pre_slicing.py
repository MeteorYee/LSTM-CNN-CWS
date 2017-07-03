# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 04/24/2017
#
# Description: some previous jobs before sentences slicing
#
# Last Modified at: 05/07/2017, by: Synrey Yee

import os

# used to store Named Entities
out_set = set([])

'''
If training materials have Named Entities(NE) labeled, this method is used to
extract them from the corpus.
This is especially for the corpus People 2014, whose NE are labeled by '[...]'.
'''
def NE_Extracting(filename):
	ipt = open(filename, 'r')
	for line in ipt:
		ustr = line.decode("utf-8")
		ustr = ustr.strip()
		lst = ustr.split()
		seeLeft = False
		NE = u""

		for token in lst:
			if len(token) < 4:
				continue

			pair = token.split(u'/')
			left = token.find(u'[')
			if left > -1:
				seeLeft = True
				NE += pair[0][left + 1 : ]

			elif seeLeft:
				NE += pair[0]
				if token.find(u']') > -1:
					seeLeft = False
					out_set.add(NE)
					NE = u""

	ipt.close()
	print "got %d Named Entities" % len(out_set)

	with open("NE_dict.txt", 'w') as opt:
		for ne in out_set:
			opt.write(ne.encode("utf-8") + '\n')


# Remove the NE labeling from People 2014 and output the results into out_list
def NE_Removing(line):
	ustr = line.decode("utf-8")
	ustr = ustr.strip()
	lst = ustr.split()
	seeLeft = False
	newline = u""

	for token in lst:
		if len(token) < 4:
			# this is a punctuation
			newline += token + u' '
			continue

		left = token.find(u'[')
		if left > -1:
			seeLeft = True
			newline += token[ : left] + token[left + 1 : ] + u' '

		elif seeLeft:
			right = token.find(u']')
			if right > -1:
				seeLeft = False
				newline += token[ : right] + u' '
			else:
				newline += token + u' '

		else:
			newline += token + u' '

	newline = newline.encode("utf-8") + '\n'
	return newline

def labeling(word, tag, X, Y, vob_dict):
	if vob_dict.has_key(word):
		X.append(vob_dict[word])
	else:
		X.append(vob_dict[u"<UNK>"])
	Y.append(tag)

# label the words for NER training
def NE_labeling(line, vob_dict):
	ustr = line.decode("utf-8")
	ustr = ustr.strip()
	lst = ustr.split()
	seeLeft = False

	X = []
	Y = []

	for token in lst:
		index = token[::-1].find(u'/') + 1
		end = len(token) - index
		word = token[ : end]
		left = token.find(u'[')

		if left > -1 and len(word) > 1:
			seeLeft = True
			labeling(word, 1, X, Y, vob_dict)

		elif seeLeft:
			right = token.find(u']')
			if right > -1:
				seeLeft = False
				labeling(word, 3, X, Y, vob_dict)
			else:
				labeling(word, 2, X, Y, vob_dict)

		else:
			labeling(word, 0, X, Y, vob_dict)

	return X, Y

# label the words for POS training
def POS_labeling(line, vob_dict, pos_dict):
	ustr = line.decode("utf-8")
	ustr = ustr.strip()
	lst = ustr.split()

	X = []
	Y = []

	for token in lst:
		index = token[::-1].find(u'/') + 1
		end = len(token) - index
		word = token[ : end]
		pos = token[end + 1 : ]

		if vob_dict.has_key(word):
			X.append(vob_dict[word])
		else:
			X.append(vob_dict[u"<UNK>"])

		if pos_dict.has_key(pos):
			Y.append(pos_dict[pos])
		else:
			Y.append(pos_dict["UNK"])

	return X, Y

# undo the labeled sentences
def CleanSentence(line, tag_set, interval = u'', breakword = False):
	ustr = line.decode("utf-8")
	ustr = ustr.strip()
	lst = ustr.split()

	newline = u""
	for token in lst:
		if token[0] == u'/':
			newline += u'/' + interval
		else:
			pair = token.split(u'/')
			length = len(pair)
			if length > 1:
				tag_set.add(pair[-1])
			
			if breakword:
				for i in xrange(length - 1):
					for char in pair[i]:
						newline += char + interval

					if i < length - 2:
						newline += u'/' + interval
			else:
				newline += pair[0] + interval

	newline = newline.encode("utf-8") + '\n'
	return newline

def All2oneFile(rootDir, out_filename):
	with open(out_filename, "w") as opt:
		for dirName, subdirList, fileList in os.walk(rootDir):
			curDir = os.path.join(rootDir, dirName)
			for file in fileList:
				line_raw_list = []
				line_list = []
				if file.endswith(".txt"):
					curFile = os.path.join(curDir, file)
					inp = open(curFile, 'r')
					for line in inp:
						ustr = line.decode("utf-8")
						ustr = ustr.strip()
						opt.write(ustr.encode("utf-8") + '\n')

# NE_Extracting("gold_raw.txt")

# with open("gold_no_NE.txt", "w") as opt:
#	NE_Removing("gold_raw.txt", opt)

# CleanSentence("gold_no_NE.txt", "gold_test.txt")