# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 04/24/2017
#
# Description: slice sentences and make them acceptable for training
#
# Last Modified at: 05/10/2017, by: Synrey Yee

# import pdb

# NOTE the sequence of elements in this list, they have different prority.
STOP = [u'。', u'？', u'！', u'?', u'!', u'：', u'；', u';', u'…', u'，', u',', u'、']

# default max length of sentence
MAX_LEN = 100

# BMES
# 0-> 'S'
# 1-> 'B'
# 2-> 'M'
# 3-> 'E'
TAGS = [0, 1, 2, 3]

# find the index of a particular symbol in line, line can be 'str' or 'list'
def FindIndex(line, symbol):
	if type(line) == unicode:
		return line.find(symbol)
	else:
		ind = -1
		for i in xrange(len(line)):
			if line[i][0] == symbol:
				ind = i
				break

		return ind

# get the length of a line according to its type
def GetLength(line):
	if type(line) == unicode:
		return len(line)
	else:
		length = 0
		for token in line:
			index = token[::-1].find(u'/') + 1
			length += len(token) - index

		return length

# write a line, depends on its type
def WriteLine(line, out_list):
	if type(line) == unicode:
		out_list.append(line.encode("utf-8") + '\n')
	else:
		newline = u' '.join(line)
		out_list.append(newline.encode("utf-8") + '\n')


# Slicing a sentence like a binary tree
def BiTreeSlicing(ustr, cnt, symbols, start, out_list):
	if start == len(STOP) - 1:
		return False, cnt

	ind = 0
	newstart = 0

	for i in xrange(start, len(symbols)):
		ind = FindIndex(ustr, symbols[i]) + 1
		if ind > 1:
			newstart = i + 1
			break

	if ind == 0:
		return False, cnt

	f1 = True
	f2 = True
	left = ustr[ : ind]
	left_len = GetLength(left)
	right = ustr[ind : ]
	right_len = GetLength(right)

	if left_len > MAX_LEN - 1:
		f1, cnt = BiTreeSlicing(left, cnt, STOP, newstart, out_list)
	elif left_len > 0:
		WriteLine(left, out_list)
		cnt += 1

	if right_len > MAX_LEN:
		f2, cnt = BiTreeSlicing(right, cnt, STOP, start, out_list)
	elif right_len > 0:
		WriteLine(right, out_list)
		cnt += 1

	flag = f1 and f2
	return flag, cnt

def SliceSentence(line, out_list, cut_count, tag = False, max_len = 100):
	if len(line) == 0:
		return False

	global MAX_LEN
	MAX_LEN = max_len

	ustr = line.decode("utf-8")
	ustr = ustr.strip()
	if tag:
		ustr = ustr.split()

	cnt = 1
	length = GetLength(ustr)
	if length < 3:
		return False

	flag = True
	if length > MAX_LEN:
		cnt = 0
		try:
			flag, cnt = BiTreeSlicing(ustr, cnt, STOP, 0, out_list)
		except Exception as e:
			print e
			return False
	
	else:
		WriteLine(ustr, out_list)

	cut_count.append(cnt)
	return flag

# labeling
def Analyze(pieces, vob_dict, max_len = 100):
	global MAX_LEN
	MAX_LEN = max_len

	new_pieces = []
	for piece in pieces:
		ustr = piece.decode("utf-8").strip()
		lst = ustr.split()

		X = []
		Y = []

		for token in lst:
			index = token[::-1].find(u'/') + 1
			word = token[ : (len(token) - index)]
			leng = len(word)
			if leng > 1:
				for j, char in enumerate(word):
					if vob_dict.has_key(char):
						X.append(vob_dict[char])
					else:
						X.append(vob_dict[u"<UNK>"])

					if j == 0:
						Y.append(TAGS[1])
					elif j == leng - 1:
						Y.append(TAGS[3])
					else:
						Y.append(TAGS[2])
			else:
				if vob_dict.has_key(word):
					X.append(vob_dict[word])
				else:
					X.append(vob_dict[u"<UNK>"])

				Y.append(TAGS[0])

		length = len(X)
		if length != len(Y):
			return [], False
		if length > MAX_LEN:
			return [], False

		for _ in xrange(length, MAX_LEN):
			X.append(0)
			Y.append(0)

		strX = ' '.join(str(x) for x in X)
		strY = ' '.join(str(y) for y in Y)
		new_piece = strX + ' ' + strY + '\n'
		new_pieces.append(new_piece)

	return new_pieces, True


'''with open("pieces.txt", "w") as opt:
	cut_count = SliceSentence("gold_test.txt", opt)'''
