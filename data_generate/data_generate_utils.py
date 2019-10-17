'''
原始数据集json转换为输入输出两个文件
'''

import json
import re
from gensim import corpora
import numpy as np
import os
import random
import itertools


# 读json文件
def load_dataset(path):
    with open(path, 'r') as f:
        data = f.readlines()
        # print(len(data))
        return data  # list

def fileDev(data):
    program = []
    describe = []
    for i in data:
        prog = json.loads(i)['code'].replace('\n','$')
        des = json.loads(i)['nl']
        program.append(prog)
        describe.append(des)
    return program,describe

def write(data,filename):
    # 写之前，先检验文件是否存在，存在就删掉
    if os.path.exists(filename):
        os.remove(filename)

    # 以写的方式打开文件，如果文件不存在，就会自动创建
    file_write_obj = open(filename, 'w')
    for var in data:
        file_write_obj.writelines(var)
        file_write_obj.write('\n')
    file_write_obj.close()


'''
提取输出数据集（program）方法
每一class带领一系列method

形如
public class AgelessEntity extends CardImpl {
'''

def conclude_signature(data):
    lines = data
    temp = []
    for i in lines:
        method_line = i.split('$')[0]
        temp.append(method_line)
    temp = list(filter(None, temp))
    method = []
    for i in temp:
        if len(str(i)) >= 2:
            method.append(i)
    # len(method) = 3,000
    return method



'''
1. 清洗method文件数据
2. 分离input output class
'''

'''
1. 清洗
'''
def generate_io(data):
    lines = data
    inlist = []
    outlist = []
    method_list = []
    t = 0

    for line in lines:
        method = str(line).replace('public ', '').replace('static ','').replace('@Override ', '').replace('private ', '').replace('@Deprecated ', '').replace('final ', '').replace('\n', '').replace('@deprecated ', '').replace('protected ', '').replace('@ShortType ', '').replace('@VisibleForTesting ', '')
        method = re.sub('@(.*) ', '', method)
        input = re.findall(r' .*?\((.*?)\){', method)
        output = re.findall(r'(.*?) .*?\(.*?\){', method)
        input_class = []
        for i in str(input).split(','):
            input_class.append(i.split(' ')[0].replace('[', '').replace(']', '').replace('\'', ''))
        inlist.append(input_class)
        outlist.append(output)
        method_name = re.findall(r'.*? (.*?)\(.*?\)', method)
        if len(method_name) > 0:
            method_name[0] += str(t)
        t += 1
        # print(method_name)

        method_list.append(method_name)
    print(len(method_list))  # 3000

    # 去空
    i = 0
    pop_list = []  # to store the index of popped item
    while i < len(method_list):
        if len(inlist[i]) == 0 or len(outlist[i]) == 0 or len(method_list[i]) == 0:
            method_list.pop(i)
            inlist.pop(i)
            outlist.pop(i)
            pop_list.append(i)
            i -= 1
        i += 1
    print(len(method_list))

    return inlist, outlist, method_list, pop_list


'''
将数据集转化为cora.cite格式，
cited_paper_id \t citing_paper_id
'''


def cited2citing(input, output, method):
    row = len(method)
    # # list 2 dic
    dic = corpora.Dictionary(method)
    dic.save_as_text('./data/dic.txt')
    dic_set = dic.token2id
    # values = []
    # for word in method:
    #     values.append(dic_set[word])

    final = []
    for i in range(row):
        for ins in input[i]:
            if ins == 'int' or ins == 'char' or ins == 'long' or ins == 'float' or ins == 'double' or ins == 'boolean' \
                    or ins == "String" or ins == 'Object' or ins == 'byte':
                # print("skip")
                continue
            for j in range(row):
                # 如果输入在输出中出现
                if ins in output[j]:
                    cited_name = method[j][0]
                    citing_name = method[i][0]
                    cited_id = dic_set[cited_name]
                    citing_id = dic_set[citing_name]
                    final.append(str(cited_id)+'\t'+str(citing_id)+'\n')
        # print(i/row)
    # print("文件计算完毕")
    return final


def write_cited(data, filename):
    # 写之前，先检验文件是否存在，存在就删掉
    if os.path.exists(filename):
        os.remove(filename)

    # 以写的方式打开文件，如果文件不存在，就会自动创建
    file_write_obj = open(filename, 'w')
    for var in data:
        file_write_obj.writelines(var)
        # file_write_obj.write('\n')
    file_write_obj.close()


'''
将数据转换为content格式
'''


def content_file_generation(node_onehot_code):
    with open('./data/cited.txt', 'r') as f:
        lines = f.readlines()
        row = len(lines)

    node_list = []

    with open('./data/dic.txt', 'r') as f_useless:
        lines2 = f_useless.readlines()[1:]
        number = len(lines2)
        # print(number)

    node = []  # temp list to store order set
    for i in range(number):  # dic length
        for j in range(row):  # cited length
            # traver front node
            if str(lines2[i]).split('\t')[0] == str(lines[j].split('\t')[1].replace('\n', '')):
                node.append(lines[j].split('\t')[0].replace('\n', ''))
                if str(i+1) != str(lines[j+1].split('\t')[1]):  # the cited file is sorted with citing index
                    break  # if the latter index has been scanned not match, this node traversing finished
        node.append(lines2[i].split('\t')[0])
        # traverse behind node
        for j in range(row):
            if str(lines2[i]).split('\t')[0] == str(lines[j].split('\t')[0]):
                node.append(lines[j].split('\t')[1].replace('\n', ''))

        node_list.append(node)
        node = []
    #
    # j = 0
    # for i in range(len(node_list)):
    #     if len(node_list[i]) > 1:
    #         print(node_list[i])
    #         j+=1
    # print(j)

    one_hot = node_onehot_code

    # 写之前，先检验文件是否存在，存在就删掉
    filename2 = "./data/content.txt"
    if os.path.exists(filename2):
        os.remove(filename2)

    print(len(node_list))
    # 以写的方式打开文件，如果文件不存在，就会自动创建
    file_write_obj = open(filename2, 'w')
    isolate_list = []
    for var in range(number):
        if len(node_list[var]) == 1:  # ignore the isolated node
            isolate_list.append(var)  # append index to list, in order to ignore its matching NL in main program
            continue
        else:
            file_write_obj.write(str(lines2[var]).split('\t')[0]+"\t")  # add the node dic index
            for i in one_hot[var]:
                file_write_obj.writelines(str(int(i))+"\t")  # add the one-hot code of node
            file_write_obj.write(str(random.randint(1, 5))+'\n')  # add the class of node, temporarily generate randomly
    file_write_obj.close()
    return isolate_list


def get_node_onehot(node_input, node_output):

    class_set = set(list(itertools.chain.from_iterable(node_input+node_output)))  # 将list变为一维，同时转换为set去重
    node_token_index = dict([(cls, i) for i, cls in enumerate(class_set)])
    node_onehot_code = np.zeros(
        (len(node_input), len(class_set)), dtype='float16')

    for i in range(len(node_input)):
        for j in range(len(node_input[i])):
            if node_input[i][j] in node_token_index:
                node_onehot_code[i, node_token_index[node_input[i][j]]] = 1

    return node_onehot_code



