import numpy as np
from model import *
from data_generate.data_generate_utils import get_node_onehot
from TagGraphEmbedding.TGEmodel import return_var
from data_generate.main import get_drop_list
# from model_seq2seq import Seq2seq

# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
# config.gpu_options.allow_growth = True #allocate dynamically

# 生成真实的 input 和 output，存储于 list
def load_data(source_path):
    inputNL = []
    with open(source_path, 'r') as fs:
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


# 将整理好的 input 转换为添加了 padding 的 index 形式
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
    print(source_batch[1])
    return source_batch, max_source_len


inputNL, input_onehot, output_onehot = load_data('./data/describe.txt')

def ignore_NL(inputNL, pop_list, isolate_list):
    for i in range(len(pop_list) - 1, -1, -1):
        inputNL.pop(pop_list[i])
    for i in range(len(isolate_list)-1, -1, -1):
        inputNL.pop(isolate_list[i])

    return inputNL


pop_list, isolate_list = get_drop_list()
inputNL = ignore_NL(inputNL, pop_list, isolate_list)

print("inputNL")
source_batch, max_source_len = add_padding(inputNL)

encoder_input_data = source_batch

node_num = len(inputNL)
max_col = len(input_onehot[0])
code_length = len(input_onehot[0][0])



print("Build Model")
model = build_test_model(max_source_len, node_num, max_col, code_length)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
# encoder_input_data = np.array(encoder_input_data).reshape(len(encoder_input_data),-1)
model.fit([np.array(encoder_input_data).reshape(len(encoder_input_data), max_source_len, 1),
           np.array(input_onehot).reshape(node_num, max_col, code_length)],
          np.array(output_onehot).reshape(node_num, code_length),
          batch_size=128,
          epochs=50,
          validation_split=0.2)

# Save model
model.save('s2s.h5')

preds = model.predict([np.array(encoder_input_data[0]).reshape(1, max_source_len, 1),
           np.array(input_onehot[0]).reshape(1, max_col, code_length)])

for i in preds:
    print(i)

