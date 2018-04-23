# LSTM-(CNN)-CRF for CWS
[![Python 2&3](https://img.shields.io/badge/python-2&3-brightgreen.svg)](https://www.python.org/) 
[![Tensorflow-1.7](https://img.shields.io/badge/tensorflow-1.7-orange.svg)](https://www.tensorflow.org/)<br>
Bi-LSTM+CNN+CRF for Chinese word segmentation. <br><br>
The **new version** has come. However, the old version is still available on another branch.

## Usage
***What's new?***
* The new system is arranged **more orderly**;
* The CNN model has been tweaked;
* Remove the limitation of maximum length of sentences, although you can still set it;
* Add gradient clipping;
* Temporally remove the pre-training (may add back in the future).
### Command Step by Step
* Preprocessing <br>
    Used to generate training files from the Corpora such as [**People 2014**](http://www.all-terms.com/bbs/thread-7977-1-1.html) and [**icwb2-data**](http://sighan.cs.uchicago.edu/bakeoff2005/). See the source code or run *python preprocess.py -h* to see more details.<br>

    For example, for the *People* data, use the default arguments;<br>

    For the icwb2-data such as PKU:<br>
    *python3 preprocess.py --all_corpora /home/synrey/data/icwb2-data/training/pku_training.utf8 --vob_path /home/synrey/data/icwb2-data/data-pku/vocab.txt --train_file_pre /home/synrey/data/icwb2-data/data-pku/train --eval_file_pre /home/synrey/data/icwb2-data/data-pku/eval --gold_file /home/synrey/data/icwb2-data/gold/pku_test_gold.utf8 --is_people False --word_freq 2*

* Training <br>
    For example:<br>
    
    *python3 -m sycws.sycws --train_prefix /home/synrey/data/cws-v2-data/train --eval_prefix /home/synrey/data/cws-v2-data/eval --vocab_file /home/synrey/data/cws-v2-data/vocab.txt --out_dir /home/synrey/data/cws-v2-data/model --model CNN-CRF*
    
    See the source code for more args' configuration. It shuold perform well with the default parameters. Naturally, you may also try out other parameter settings.
    
## About the Models
### Bi-LSTM-SL-CRF 
Take reference to [Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. Neural Architectures for Named Entity Recognition. In Proc. ACL. 2016.](http://www.aclweb.org/anthology/N16-1030)<br><br>
Actually, there is a *single layer* (SL) between BiLSTM and CRF.

### Bi-LSTM-CNN-CRF
See [Here](http://htmlpreview.github.io/?https://github.com/MeteorYee/LSTM-CNN-CWS/blob/master/Extra/Bi-LSTM_CNN.html).<br>
Namely, the single layer between BiLSTM and CRF is replaced by a layer of CNN.
    
### Comparison
Experiments on corpus [**People 2014**](http://www.all-terms.com/bbs/thread-7977-1-1.html).

|     Models    |  Bi-LSTM-SL-CRF  |  Bi-LSTM-CNN-CRF  |
| :-----------: | :--------------: | :---------------: |
|   Precision   |     96.25%       |      96.30%       |
|     Recall    |     95.34%       |      95.70%       |
|     F-value   |     95.79%       |    **96.00%**     |

## Segmentation
* Inference <br>
For example, to use model **BiLSTM-CNN-CRF** for decoding.<br>

    *python3 -m sycws.sycws --train_prefix /home/synrey/data/cws-v2-data/train --eval_prefix /home/synrey/data/cws-v2-data/eval --vocab_file /home/synrey/data/cws-v2-data/vocab.txt --out_dir /home/synrey/data/cws-v2-data/model --inference_input_file /home/synrey/data/cws-v2-data/test.txt --inference_output_file /home/synrey/data/cws-v2-data/result.txt*
    
    Set *--model CRF* to use model **BiLSTM-SL-CRF** for inference.
    
* PRF Scoring <br>
    
    *python3 PRF_Score.py <test_file> <gold_file>*
