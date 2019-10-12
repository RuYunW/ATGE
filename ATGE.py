import tensorflow.compat.v1 as tf
import numpy as np
import random
import time
from model import *
from node_onehot_generate import get_node_onehot

# from model_seq2seq import Seq2seq

# 生成真实的 input 和 output，存储于 list
def load_data(source_path,target_path):
    docs_source = []
    docs_target = []
    with open(source_path,'r') as fs:
        lines = fs.readlines()
        for line in lines[:1000]:
            docs_source.append(line)

    with open(target_path,'r') as ft:
        lines = ft.readlines()
        for line in lines[:1000]:
            docs_target.append(line)

    return docs_source, docs_target


# 生成 doc 中所有词的词汇表
# w2i 为 word → index，i2w 为 index → word，
# 二者相互匹配
def make_vocab(docs):
    w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2}
    i2w = {0: "_PAD", 1: "_GO", 2: "_EOS"}
    for doc in docs:
        for w in doc:
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
    return w2i, i2w

# 将 doc 中每一句话，转换为 index list 的形式，如 [1, 11, 5, 14, 2, 0, 0, 0, 0, 0]
def doc_to_seq(docs):
    w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2}
    i2w = {0: "_PAD", 1: "_GO", 2: "_EOS"}
    seqs = []
    for doc in docs:
        seq = []
        for w in doc:
            if w not in w2i:
                i2w[len(w2i)] = w  # 相当于再次制作单词表
                w2i[w] = len(w2i)
            seq.append(w2i[w])
        seqs.append(seq)
    return seqs, w2i, i2w


# 将 整理好的 input 和 output 转换为添加了 padding 的 index 形式
def add_padding(docs_source, docs_target):
    # 生成单词表
    w2i_source, i2w_source = make_vocab(docs_source)
    w2i_target, i2w_target = make_vocab(docs_target)

    # 寻找最大长度
    source_lens = [len(i) for i in docs_source]
    target_lens = [len(i)+2 for i in docs_target]
    max_source_len = max(source_lens)
    max_target_len = max(target_lens)

    source_batch = []
    target_batch = []
    # 添加 padding
    for i in range(len(docs_source)):
        source_seq = [w2i_source[w] for w in docs_source[i]] + [w2i_source["_PAD"]] * (
                    max_source_len - len(docs_source[i]))
        target_seq = [w2i_target["_GO"]]+[w2i_target[w] for w in docs_target[i]] + [w2i_target["_EOS"]] + [w2i_target["_PAD"]] * (
                    max_target_len - 2 - len(docs_target[i]))
        source_batch.append(source_seq)
        target_batch.append(target_seq)

    return source_batch, target_batch, max_source_len, max_target_len


docs_source, docs_target = load_data('./data/describe.txt', './data/all_method.txt')
source_batch, target_batch, max_source_len, max_target_len = add_padding(docs_source,docs_target)

encoder_input_data = source_batch
decoder_target_data = target_batch
decoder_input_data = []
for i in target_batch:
    i.remove(1)
    i.remove(2)
    decoder_input_data.append(i)

node_onthot = get_node_onehot('./data/method_io.xlsx')  # 获取节点 one-hot 编码
model = build_test_model(max_source_len, max_target_len, len(encoder_input_data), len(decoder_input_data))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
# encoder_input_data = np.array(encoder_input_data).reshape(len(encoder_input_data),-1)
model.fit(np.array(encoder_input_data).reshape(len(encoder_input_data), 1, max_source_len),
          np.array(decoder_target_data).reshape(len(decoder_input_data), 1, max_target_len-2),
          batch_size=64,
          epochs=10,
          validation_split=0.2)

# Save model
model.save('s2s.h5')



