# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 05/20/2018
#
# Description: The PRF scoring script used for evaluation
#
# Last Modified at: 05/20/2018, by: Synrey Yee

'''
==========================================================================
  Copyright 2018 Xingyu Yi (Alias: Synrey Yee) All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
==========================================================================
'''

from __future__ import print_function
from __future__ import division

import codecs

def get_prf_score(test_list, gold_file):
  e = 0 # wrong words number
  c = 0 # correct words number
  N = 0 # gold words number
  TN = 0 # test words number

  assert type(test_list) == list
  test_raw = []
  for line in test_list:
    sent =  line.strip().split()
    if sent:
      test_raw.append(sent)

  gold_raw = []
  with codecs.open(gold_file, 'r', "utf-8") as inpt2:
    for line in inpt2:
      sent = line.strip().split()
      if sent:
        gold_raw.append(sent)
        N += len(sent)

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
            print ("Line: %d" % (i + 1))
            print ("\nIt is the user's responsibility that a sentence in <test file> must", end = ' ')
            print ("have the SAME LENGTH with its corresponding sentence in <gold file>.\n")
            raise e
          
      ig += 1
      it += 1

    TN += len(test_sent)

  e = TN - c
  precision = c / TN
  recall = c / N
  F = 2 * precision * recall / (precision + recall)
  error_rate = e / N

  print ("Correct words: %d"%c)
  print ("Error words: %d"%e)
  print ("Gold words: %d\n"%N)
  print ("precision: %f"%precision)
  print ("recall: %f"%recall)
  print ("F-Value: %f"%F)
  print ("error_rate: %f"%error_rate)

  return F