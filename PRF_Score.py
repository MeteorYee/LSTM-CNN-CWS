#-*- coding: utf-8 -*-

# Perceptron word segment for Chinese sentences
# 
# File usage:
# Results score. PRF
# 
# Author:
# Synrey Yee

from __future__ import division
import sys
# import pdb

e = 0 # wrong words number
c = 0 # correct words number
N = 0 # gold words number
TN = 0 # test words number

testfile = sys.argv[1]
goldfile = sys.argv[2]

inpt1 = open(testfile, 'rU')
inpt2 = open(goldfile, 'rU')

test_raw = []

for ind, line in enumerate(inpt1):
    sent = []

    for word in line.decode("utf-8").split():
        sent.append(word)

    test_raw.append(sent)

gold_raw = []

for ind, line in enumerate(inpt2):
    sent = []

    for word in line.decode("utf-8").split():
        sent.append(word)
        N += 1

    gold_raw.append(sent)

for i, gold_sent in enumerate(gold_raw):
    test_sent = test_raw[i]

    ig = 0
    it = 0
    glen = len(gold_sent)
    tlen = len(test_sent)
    while True:
        if ig >= glen or it >= tlen:
            break

        gword = gold_sent[ig]
        tword = test_sent[it]
        if gword == tword:
            c += 1
        else:
            lg = len(gword)
            lt = len(tword)
            while lg != lt:
                try:
                    if lg < lt:
                        ig += 1
                        gword = gold_sent[ig]
                        lg += len(gword)
                    else:
                        it += 1
                        tword = test_sent[it]
                        lt += len(tword)
                except Exception as e:
                    # pdb.set_trace()
                    print "Line: %d" % (i + 1)
                    print "\nIt is the user's responsibility that a sentence in <test file> must",
                    print "have a SAME LENGTH with its corresponding sentence in <gold file>.\n"
                    raise e
                
        ig += 1
        it += 1

    TN += len(test_sent)

e = TN - c
precision = c / TN
recall = c / N
F = 2 * precision * recall / (precision + recall)
error_rate = e / N

print "Correct words: %d"%c
print "Error words: %d"%e
print "Gold words: %d"%N
print
print "precision: %f"%precision
print "recall: %f"%recall
print "F-Value: %f"%F
print "error_rate: %f"%error_rate