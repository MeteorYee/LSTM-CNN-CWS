#!/usr/bin/env bash
#
# Author: Synrey Yee
#
# Created at: 07/10/2017
#
# Description: training script
#
# Last Modified at: 07/10/2017, by: Synrey Yee

python preprocess.py --rootDir $1 --corpusAll Corpora/people2014All.txt --resultFile pre_chars_for_w2v.txt

./third_party/word2vec -train pre_chars_for_w2v.txt -save-vocab pre_vocab.txt -min-count 3

python SentHandler/replace_unk.py pre_vocab.txt pre_chars_for_w2v.txt chars_for_w2v.txt

./third_party/word2vec \
-train chars_for_w2v.txt -output char_vec.txt -size 50 -sample 1e-4 -negative 0 -hs 1 -binary 0 -iter 5

python pre_train.py --corpusAll Corpora/people2014All.txt --vecpath char_vec.txt \
--train_file Corpora/train.txt --test_file Corpora/test.txt

# python ./CWSTrain/lstm_crf_train.py --train_data_path Corpora/train.txt --test_data_path Corpora/test.txt \
# --word2vec_path char_vec.txt

python ./CWSTrain/lstm_cnn_train.py --train_data_path Corpora/train.txt --test_data_path Corpora/test.txt \
--word2vec_path char_vec.txt