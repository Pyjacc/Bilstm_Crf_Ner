'''
使用传统方法（双向LSTM + CRF）实现序列标注（NER）

参考github开源代码（采用预训练的方式进行ner）：
https://github.com/bojone/bert4keras/blob/master/examples/task_sequence_labeling_ner_crf.py

说明：tensorflow版本为1.14版本
'''

import numpy as np
import os, time, sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf     # 使用tensorflow 1.14版本
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from test_data import  batch_yield_test,read_corpus_test

learning_rate = 0.001
decay_rate = 0.8

# 一层神经网络 + 一层CRF
class BiLSTM_CRF(object):
    def __init__(self, embeddings, tag2label, vocab,batch_size,epoch,hidden_dim,CRF,update_embedding,shuffle):
        self.batch_size = batch_size
        self.epoch_num = epoch
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        self.CRF = CRF      # 是否使用CRF
        self.update_embedding = update_embedding    # 是否更新embedding
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = shuffle

    # 构建网络
    def build_graph(self):
        self.add_placeholders()     # 站位符
        self.lookup_layer_op()      # 构建embedding
        self.biLSTM_layer_op()      # LSTM层
        self.loss_op()              # 损失函数(CRF在这一层中)
        self.trainstep_op()         # 训练
        self.init_op()              # 初始化变量
        #for test
        self.softmax_pred_op()      # 预测概率

    def add_placeholders(self):
        # shape=[None,None]：None为展位符，第一个None位置为batch_siez维度，第二个None位置为seq_len维度
        self.word_ids = tf.placeholder(tf.int32,shape=[None,None], name="word_ids")
        self.labels = tf.placeholder(tf.int32,shape=[None,None],name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32,shape=[None],name="sequence_lengths")
        self.lr_pl = tf.Variable(0.0,trainable=False)       #学习率

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embedding = tf.Variable(self.embeddings,
                                          dtype=tf.float32,
                                          trainable=self.update_embedding,
                                          name="_word_embeddings")
            self.word_embedding = tf.nn.embedding_lookup(params=_word_embedding,
                                                         ids=self.word_ids,
                                                         name="word_embeddings")

    def biLSTM_layer_op(self):
        cell_fw = LSTMCell(self.hidden_dim)     # 前向LSTM
        cell_bw = LSTMCell(self.hidden_dim)     # 后向LSTM
        # 输出有2个，一个前向输出，一个后向输出
        (output_fw_seq,output_bw_seq),_ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = cell_fw,
            cell_bw = cell_bw,
            inputs = self.word_embedding,
            sequence_length = self.sequence_lengths,
            dtype = tf.float32
        )
        output = tf.concat([output_fw_seq,output_bw_seq],axis=-1)   # axis=-1,拼接词向量部分
        W = tf.get_variable(name="W",
                            shape=[2 * self.hidden_dim,self.num_tags],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32)
        b = tf.get_variable(name="b",
                            shape=[self.num_tags],
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float32)

        s = tf.shape(output)
        output = tf.reshape(output,[-1,2*self.hidden_dim])
        pred = tf.matmul(output,W) + b
        self.logits = tf.reshape(pred,[-1,s[1],self.num_tags])      # 没有经过激活函数


    def loss_op(self):
        if self.CRF:        # 如果使用CRF
            log_likelihood,self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                       tag_indices=self.labels,
                                                                       sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)
        else:   # 如果不使用CRF，使用传统的交叉熵的方式。
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses,mask)   #loss乘以mask矩阵，去除掉mask中为0部分的损失
            self.loss = tf.reduce_mean(losses)

    def trainstep_op(self):
        optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl).minimize(self.loss)    # 优化器
        self.train_op = optim

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def softmax_pred_op(self):
       if not self.CRF:
           self.labels_softmax_ = tf.argmax(self.logits,axis=-1)            #拿到概率最大的label
           self.labels_softmax_ = tf.cast(self.labels_softmax_,tf.int32)    # 转为int型数据格式

    def train(self, train):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(self.init_op)
            for epoch in range(self.epoch_num):
                sess.run(tf.assign(self.lr_pl,learning_rate*(decay_rate ** epoch)))
                self.run_one_epoch(sess,train,epoch)
            saver.save(sess,'model/ner.model')

    def run_one_epoch(self, sess, train, epoch):
        batches = batch_yield(train,self.batch_size,self.vocab,self.tag2label,shuffle=self.shuffle)
        for step,(seqs,labels) in enumerate(batches):
            feed_dict ,_= self.get_feed_dict(seqs,labels)
            _,loss_train = sess.run([self.train_op,self.loss],feed_dict=feed_dict)
            print('epoch {}, step {}, loss {:.4}'.format(epoch+1,step+1,loss_train))


    def get_feed_dict(self, seqs, labels):
        word_ids,seq_len_list = pad_sequences(seqs,pad_mark=0)
        feed_dict = {self.word_ids:word_ids,self.sequence_lengths:seq_len_list}
        if labels is not  None:
            labels_,_ = pad_sequences(labels,pad_mark=0)
            feed_dict[self.labels] = labels_
        return  feed_dict,seq_len_list

    def test(self,test_data):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,'model/ner.model')
            tag,samples = self.demo_one(sess,test_data)
            self.display(tag,samples)

    def display(self,tag,samples):
        print(tag)
        print(samples)
        site = []
        surgery = []
        drug = []
        Independent = []
        symptom = []
        h = False
        for i ,x in enumerate(tag):
            if 1 == int(x):
                if h:
                    if len(site) > 0:
                        site[-1].append(i)
                else:
                    site.append([i])
                    h = True
            else:
                h = False
        h = False
        for i, x in enumerate(tag):
            if 2 == int(x):
                if h:
                    if len(surgery) > 0:
                        surgery[-1].append(i)
                else:
                    surgery.append([i])
                    h = True
            else:
                h = False
        h = False
        for i, x in enumerate(tag):
            if 3 == int(x):
                if h:
                    if len(drug) > 0:
                        drug[-1].append(i)
                else:
                    drug.append([i])
                    h = True
            else:
                h = False
        h = False
        for i, x in enumerate(tag):
            if 4 == int(x):
                if h:
                    if len(Independent) > 0:
                        Independent[-1].append(i)
                else:
                    Independent.append([i])
                    h = True
            else:
                h = False
        h = False
        for i, x in enumerate(tag):
            if 5 == int(x):
                if h:
                    if len(symptom) > 0:
                        symptom[-1].append(i)
                else:
                    symptom.append([i])
                    h = True
            else:
                h = False
        site_ = []
        symptom_ = []
        Independent_ = []
        drug_ = []
        surgery_ = []
        for l in site:
            one = []
            for index in l:
                one.append(samples[index])
            one = ''.join(one)
            site_.append(one)
        for l in symptom:
            one = []
            for index in l:
                one.append(samples[index])
            one = ''.join(one)
            symptom_.append(one)
        for l in Independent:
            one = []
            for index in l:
                one.append(samples[index])
            one = ''.join(one)
            Independent_.append(one)
        for l in drug:
            one = []
            for index in l:
                one.append(samples[index])
            one = ''.join(one)
            drug_.append(one)
        for l in surgery:
            one = []
            for index in l:
                one.append(samples[index])
            one = ''.join(one)
            surgery_.append(one)

        order = {}
        for q in range(len(site_)):
            siteindex = site[q]
            if len(siteindex) > 1:
                start_num = siteindex[0]
                end_num = siteindex[-1]+1
            else:
                start_num = siteindex[0]
                end_num = start_num + 1

            w_site = site_[q] + '\t' + str(start_num) + '\t' + str(end_num) +  '\t' + '解剖部位' + ';'
            order[start_num] = w_site

        for q1 in range(len(surgery_)):
            surg_index = surgery[q1]
            if len(surg_index) > 1:
                start_num = surg_index[0]
                end_num = surg_index[-1]+1
            else:
                start_num = surg_index[0]
                end_num = start_num + 1
            w_surgery = surgery_[q1] + '\t' + str(start_num)+'\t'+str(end_num)+'\t'+'手术'+';'
            order[start_num] = w_surgery
        for q2 in range(len(drug_)):
            drugindex = drug[q2]
            if len(drugindex) > 1:
                start_num = drugindex[0]
                end_num = drugindex[-1]+1
            else:
                start_num = drugindex[0]
                end_num = start_num + 1
            w_drug = drug_[q2] + '\t' + str(start_num)+'\t'+str(end_num)+'\t'+'药物'+';'
            order[start_num] = w_drug
        for q3 in range(len(Independent_)):
            Independentindex = Independent[q3]
            if len(Independentindex) > 1:
                start_num = Independentindex[0]
                end_num = Independentindex[-1]+1
            else:
                start_num = Independentindex[0]
                end_num = start_num + 1
            w_Independent = Independent_[q3] +'\t' + str(start_num)+'\t'+str(end_num)+'\t'+'独立症状'+';'
            order[start_num] = w_Independent
        for q4 in range(len(symptom_)):
            symptomindex = symptom[q4]
            if len(symptomindex) > 1:
                start_num = symptomindex[0]
                end_num = symptomindex[-1]+1
            else:
                start_num = symptomindex[0]
                end_num = start_num + 1
            w_symptom = symptom_[q4] + '\t' + str(start_num)+'\t'+str(end_num)+'\t'+'症状描述'+';'
            order[start_num] = w_symptom
        index_order = []
        for index in order.keys():
            index_order.append(index)
        index_order.sort()

        print('测试样本为：', ''.join(samples))
        print('识别出来的解剖部位为：',site_)
        print('识别出来的手术为：',surgery_)
        print('识别出来的药物为：',drug_)
        print('识别出来的独立症状为：',Independent_)
        print('识别出来的症状描述为：',symptom_)

    # 取一个
    def demo_one(self, sess,test_data):
        seqs, sample = batch_yield_test(test_data, self.vocab)
        label_list_, _ = self.predict_one_batch(sess, seqs)
        pre_label = []
        for label in label_list_:
            print(label)
            pre_label.append(label[0])
        return pre_label,sample


    def predict_one_batch(self, sess, seqs):
        feed_dict, seq_len_list = self.get_feed_dict(seqs,None)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list