# LSTM-CNN-CWS
Bi-LSTM+CNN for Chinese word segmentation<br>
The new version has come! See [here](https://github.com/MeteorYee/LSTM-CNN-CWS)

## Acknowledgement
The implementation of this repository partly refers to [Koth's kcws](https://github.com/koth/kcws).

## Usage
Have tensorflow 1.2 installed.
### Command Step by Step
* Preprocessing <br>
    
    *python preprocess.py --rootDir \<ROOTDIR> --corpusAll Corpora/people2014All.txt --resultFile pre_chars_for_w2v.txt*
    
    ROOTDIR is the absolute path of your corpus. Run *python preprocess.py -h* to see more details.
    
* Word2vec Training <br>
    
    *./third_party/word2vec -train pre_chars_for_w2v.txt -save-vocab pre_vocab.txt -min-count 3*
    
    *python SentHandler/replace_unk.py pre_vocab.txt pre_chars_for_w2v.txt chars_for_w2v.txt*
    
    *./third_party/word2vec -train chars_for_w2v.txt -output char_vec.txt \\<br>
    -size 50 -sample 1e-4 -negative 0 -hs 1 -binary 0 -iter 5*
    
    First off, the file **word2vec.c** in third_party directory should be compiled (see third_party/compile_w2v.sh). Then word2vec counts the characters which have a frequency more than 3 and saves them into file **pre_vocab.txt**. After replacing with **"UNK"** the words that are not in pre_vocab.txt, finally, word2vec training begins.
    
* Generate Training Files <br>
    
    *python pre_train.py --corpusAll Corpora/people2014All.txt --vecpath char_vec.txt \\<br>
    --train_file Corpora/train.txt --test_file Corpora/test.txt*
    
    Run *python pre_train.py -h* to see more details.
    
* Training <br>
    
    *python ./CWSTrain/lstm_cnn_train.py --train_data_path Corpora/train.txt \\<br>
    --test_data_path Corpora/test.txt --word2vec_path char_vec.txt*
    
    Arguments of *lstm_cnn_train.py* are set by **tf.app.flags**. See the file for more args' configurations.
    
### One-step Training
    
    ./cws_train.sh <ROOTDIR>
    
## About the Models
### Bi-LSTM-CRF
Take reference to [Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. Neural Architectures for Named Entity Recognition. In Proc. ACL. 2016.](http://www.aclweb.org/anthology/N16-1030)
* Freeze graph <br>

    *python tools/freeze_graph.py --input_graph Logs/seg_logs/graph.pbtxt --input_checkpoint Logs/seg_logs/model.ckpt --output_node_names "input_placeholder, transitions, Reshape_7" --output_graph Models/lstm_crf_model.pbtxt*

    Build model for segmentation.
### Bi-LSTM-CNN
See [Here](http://htmlpreview.github.io/?https://github.com/MeteorYee/LSTM-CNN-CWS/blob/master/Extra/Bi-LSTM_CNN.html).
* Freeze graph <br>

    *python tools/freeze_graph.py --input_graph Logs/seg_cnn_logs/graph.pbtxt --input_checkpoint Logs/seg_cnn_logs/model.ckpt --output_node_names "input_placeholder,Reshape_5" --output_graph Models/lstm_cnn_model.pbtxt*
    
### Comparison
Experiments on corpus [**People 2014**](http://www.all-terms.com/bbs/thread-7977-1-1.html).

|     Models    |  Bi-LSTM-CRF  |  Bi-LSTM-CNN  |
| ------------- | ------------- | ------------- |
|   Precision   |     96.11%    |     96.27%    |
|     Recall    |     95.73%    |     96.34%    |
|    F-value    |     95.92%    |     **96.30%**    |

## Segmentation
* Dump Vocabulary <br>

    *python tools/vob_dump.py --vecpath char_vec.txt --dump_path Models/vob_dump.pk* <br>

    This step is **neccessary** for the seg model.

* Seg Script <br>

    Use file **tools/crf_seg.py** and file **tools/cnn_seg.py**. You may refer to the files about detailed parameters config. <br>
    For default, at the root path of this repository, *python tools/crf_seg.py* or *python tools/cnn_seg.py* will work.
    
* PRF Scoring <br>
    
    *python PRF_Score.py Results/cnn_result.txt Corpora/test_gold.txt*
    
    Result files are put in directory **Results/**.
