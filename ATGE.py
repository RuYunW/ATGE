import tensorflow.compat.v1 as tf
import numpy as np
import random
import time
from model_seq2seq_contrib import Seq2seq

# from model_seq2seq import Seq2seq





def load_data(path):
    num2en = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight",
              "9": "nine", "10": "ten", "11": "eleven", "12": "twelve", "0": "zero"}
    docs_source = []
    docs_target = []
    for i in range(10000):
        doc_len = random.randint(1, 8)
        doc_source = []
        doc_target = []
        for j in range(doc_len):
            num = str(random.randint(0, 12))
            doc_source.append(num)
            num = str(random.randint(0, 12))
            doc_target.append(num2en[num])
        docs_source.append(doc_source)
        docs_target.append(doc_target)

    return docs_source, docs_target


def make_vocab(docs):
    w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2}
    i2w = {0: "_PAD", 1: "_GO", 2: "_EOS"}
    for doc in docs:
        for w in doc:
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
    return w2i, i2w


def doc_to_seq(docs):
    w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2}
    i2w = {0: "_PAD", 1: "_GO", 2: "_EOS"}
    seqs = []
    for doc in docs:
        seq = []
        for w in doc:
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            seq.append(w2i[w])
        seqs.append(seq)
    return seqs, w2i, i2w


def get_batch(docs_source, w2i_source, docs_target, w2i_target, batch_size):
    ps = []
    while len(ps) < batch_size:
        ps.append(random.randint(0, len(docs_source) - 1))

    source_batch = []
    target_batch = []

    source_lens = [len(docs_source[p]) for p in ps]
    target_lens = [len(docs_target[p]) + 1 for p in ps]

    max_source_len = max(source_lens)
    max_target_len = max(target_lens)

    for p in ps:
        source_seq = [w2i_source[w] for w in docs_source[p]] + [w2i_source["_PAD"]] * (
                    max_source_len - len(docs_source[p]))
        target_seq = [w2i_target["_GO"]]+[w2i_target[w] for w in docs_target[p]] + [w2i_target["_EOS"]] + [w2i_target["_PAD"]] * (
                    max_target_len - 1 - len(docs_target[p]))
        source_batch.append(source_seq)
        target_batch.append(target_seq)

    return source_batch, source_lens, target_batch, target_lens

docs_source, docs_target = load_data('')

w2i_source, i2w_source = make_vocab(docs_source)

w2i_target, i2w_target = make_vocab(docs_target)
source_batch, source_lens, target_batch, target_lens = get_batch(docs_source, w2i_source, docs_target, w2i_target, 64)
# print(docs_source[0])
print(source_batch[0])
print(source_lens)

print(target_batch[0])
# print(len(target_batch))