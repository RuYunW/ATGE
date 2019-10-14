'''
1. 清洗method文件数据
2. 分离input output class
'''

import os
import re
from openpyxl import load_workbook, Workbook
'''
1. 清洗
'''
def generate_io(data):
    lines = data
    inlist = []
    outlist = []
    method_list = []

    for line in lines:
        method = str(line).replace('public ','').replace('static ','').replace('@Override ','').replace('private ','').replace('@Deprecated ','').replace('final ','').replace('\n','').replace('@deprecated ','').replace('protected ','').replace('@ShortType ','').replace('@VisibleForTesting ','')
        method = re.sub('@(.*) ','',method)
        input = re.findall(r' .*?\((.*?)\){',method)
        output = re.findall(r'(.*?) .*?\(.*?\){',method)
        input_class = []
        for i in str(input).split(','):
            input_class.append(i.split(' ')[0].replace('[','').replace(']','').replace('\'',''))
        inlist.append(input_class)
        outlist.append(output)
        method_name = re.findall(r'.*? (.*?)\(.*?\)',method)
        method_list.append(method_name)

    # 去空
    for i in range(len(method_list)):
        if inlist[i] == '' or outlist == '':
            method_list.remove(method_list[i])
            inlist.remove(inlist[i])
            outlist.remove(outlist[i])

    return inlist,outlist,method_list


