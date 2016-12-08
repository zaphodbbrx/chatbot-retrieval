## Retrieval-Based Conversational Model in Tensorflow (Ubuntu Dialog Corpus)

#### [Please read the blog post for this code](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow)

#### Overview

The code here implements the Dual LSTM Encoder model from [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](http://arxiv.org/abs/1506.08909).

#### Setup

This code uses Python 3. Clone the repository and install all required packages:

```
1. install tensorflow (version 0.11 and above wokr correctly, version 0.10 not tested)
2. (optional) install cuda + cudnn (for gpu support)
2. pip3 install -U pip
3. pip3 install -r requirements.txt
```

#### Get the Data


Download the train/dev/test data [here](https://drive.google.com/open?id=0B_bZck-ksdkpVEtVc1R6Y01HMWM) and extract the acrhive into `./data`.


#### Training

```
python3 udc_train.py
or------------------
sh train.sh
```


#### Evaluation

```
python3 udc_test.py --model_dir=...
or------------------
sh test.sh
```


**example:**
```
python3 udc_test.py --model_dir=./runs/1481183770/
or------------------
sh predict.sh
```

#### Evaluation

```
python3 udc_predict.py --model_dir=...
```

**example:**
```
python3 udc_predict.py --model_dir=./runs/1481183770/
```

#### Issues

* if you have problem's with loading **CUDA library libcuda.so.1** use *.sh script, or export _variables_ in bash:

```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
```

* if you have multiple gpu devices and expecting troubles with performance, manualy select device in bash:

```
export CUDA_VISIBLE_DEVICES=0
```

* if you have error (see below), you maybe use trained model from other machine, and you must retrain model on own machine

>tensorflow.python.framework.errors.NotFoundError: /home/user/git/chatbot/chatbot-retrieval/runs/1481104318

