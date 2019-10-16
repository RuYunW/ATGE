import numpy as np
from model import *
from data_generate.data_generate_utils import get_node_onehot
from TagGraphEmbedding.TGEmodel import return_var
# from model_seq2seq import Seq2seq

# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
# config.gpu_options.allow_growth = True #allocate dynamically

# 生成真实的 input 和 output，存储于 list
def load_data(source_path):
    inputNL = []
    with open(source_path,'r') as fs:
        lines = fs.readlines()
        for line in lines:
            inputNL.append(line)
    input_onehot, output_onehot = return_var()

    return inputNL, input_onehot, output_onehot


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


# 将 整理好的 input 转换为添加了 padding 的 index 形式
def add_padding(docs_source):
    # 生成单词表
    w2i_source, i2w_source = make_vocab(docs_source)
    # 寻找最大长度
    source_lens = [len(i) for i in docs_source]
    max_source_len = max(source_lens)
    source_batch = []
    # 添加 padding
    for i in range(len(docs_source)):
        source_seq = [w2i_source[w] for w in docs_source[i]] + [w2i_source["_PAD"]] * (
                    max_source_len - len(docs_source[i]))
        source_batch.append(source_seq)

    return source_batch, max_source_len


inputNL, input_onehot, output_onehot = load_data('./data/describe.txt')
source_batch, max_source_len = add_padding(inputNL)

encoder_input_data = source_batch

print(len(inputNL))
print(len(output_onehot))
print(len(input_onehot))

exit()

node_onehot = get_node_onehot('./data/method_io.xlsx')  # 获取节点 one-hot 编码
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



