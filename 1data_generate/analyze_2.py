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

    return method



