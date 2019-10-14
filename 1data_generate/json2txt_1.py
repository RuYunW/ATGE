'''
原始数据集json转换为输入输出两个文件
'''

import json
import os

# 读json文件
def load_dataset(path):
    with open(path,'r') as f:
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

# data = load_dataset('./data/test.json')  # list
# program,describe = fileDev(data)
# write(program,"./data/program.txt")
# write(describe,"./data/describe.txt")
# print("完毕")