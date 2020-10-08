import tensorflow as tf
import numpy as np
import os
import os, argparse, time, random
from model import BiLSTM_CRF
from data import read_corpus, read_dictionary, tag2label, random_embedding


## Session configuration
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
# config = tf.ConfigProto()

## hyperparameters

embedding_dim = 128
tag2label = {"N": 0,
             "解剖部位": 1,
             "手术": 2,
             "药物": 3,
             "独立症状": 4,
             "症状描述": 5}
## get char embeddings
word2id = read_dictionary('./vocab.pkl')
embeddings = random_embedding(word2id,embedding_dim)
train_data = read_corpus('./c.txt')
model = BiLSTM_CRF(embeddings,tag2label,word2id,2,2,128,True,True,True)
model.build_graph()

# train
model.train(train_data)

# test（效果比较差）
model.test("患者于半月前五明显诱因出现进食后中上腹不适，"
           "每次持续数分钟自行缓解，无恶心，呕吐，反酸"
           "小便正常，体重无明显变化。")