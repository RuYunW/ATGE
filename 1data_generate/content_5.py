'''
将数据转换为content格式
'''

import numpy as np
import os

def content_file_generation(node_onehot_code):
    with open('../data/cited.txt','r') as f:
        lines = f.readlines()
        row = len(lines)

    node = []
    node_list = []
    # for i in range(row):
    #     ins,outs = lines[i].split('\t')

    with open('../data/dic.txt','r') as f_useless:
        lines2 = f_useless.readlines()
        number = len(lines2)

    #     # 24928

    for i in range(number):
        for j in range(len(lines)):
            # input
            if str(i+1) == str(lines[j].split('\t')[1]):
                node.append(lines[j].split('\t')[0].replace('\n',''))
                if str(i+1) != str(lines[j+1].split('\t')[1]):
                    break
        node.append(i+1)
        for j in range(len(lines)):
            if str(i+1) == str(lines[j].split('\t')[0]):
                node.append(lines[j].split('\t')[1].replace('\n',''))
        print(node)
        node_list.append(node)
        node = []

    # 生成content
    one_hot = node_onehot_code

    # 写之前，先检验文件是否存在，存在就删掉
    filename2 = "../data/content.txt"
    if os.path.exists(filename2):
        os.remove(filename2)


    # 以写的方式打开文件，如果文件不存在，就会自动创建
    file_write_obj = open(filename2, 'w')
    for var in range(len(one_hot)):
        if np.sum(one_hot[var]) == 1:
            continue
        else:
            file_write_obj.write(str(var+1)+"\t")
            for i in one_hot[var]:
                file_write_obj.writelines(str(int(i))+"\t")
            file_write_obj.write('NIL\n')
    file_write_obj.close()


