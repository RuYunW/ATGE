'''
Tag-graph embedding generation algorithm
'''

from TGEutils import *
import numpy as np
from numpy import *
from keras.layers import LSTM,Dense,Dropout
from keras.models import Sequential

# Initialize
# x,cites,content,class_set,code_length = load_data(cites_path = './data/cited.txt',content_path = './data/content.txt')
x,cites,content,class_set,code_length = load_data()

order = []  # Create an empty list
K = 2  # hops
h = [[] for i in range(K+1)]  # temp matrices

node_num = len(x[:])
# print(x[0][0])  # [[0,...,0],[...],...,[...]]

h[0]+=x[:]
y = (np.array(h[0])[:, [i for i in range(1, code_length+1)]]).tolist()
print(len(h[0]))
# front_code = front_node_garthing(x,cites,content,class_set)
# behind_code = behind_node_garthing(x,cites,content,class_set)

temp = []
emb = []
# ave_front_code = ave(front_code)
# ave_behind_code = ave(behind_code)

h_front = [[] for i in range(K+1)]
h_behind = [[] for i in range(K+1)]
# len(h[]=2708) = n(node)   len(h[][] = 1433) = n(words) -- feature matrices

___ = ave(front_node_garthing(x,cites,content,class_set))

# q = []
# qt = []
# for i in ___:
#     if len(list(i)) >=1 :
#         for j in i:
#             qt.append(j)
#
#
#     q.append(qt)
#     qt = []

max_col = max(len(j) for j in ___)
print(max_col)

zo = [0 for _ in range(code_length)]

for i in range(len(___)):
    for j in range(max_col-len(___[i])):
        ___[i].append(zo)

# print(len(___[0]))
# exit(0)
#------------------------
# h_front[0] = summ(ave(front_node_garthing(x,cites,content,class_set)),code_length)
# h_behind[0] = summ(ave(behind_node_garthing(x,cites,content,class_set)),code_length)
#
# # h = ave(h_front_cal(x,cites,content,class_set,h_front))
#
order_set = []
M = len(x[:])  # 2708
#
# # h_sum = sum(h_front)
# for k in range(1,K+1):  # hop数  K
#     h_front[k] = summ(ave(h_front_cal(x, cites, content, class_set,h_front[k-1])),code_length)
#     h_behind[k] = summ(ave(h_behind_cal(x, cites, content, class_set,h_behind[k-1])),code_length)
#     temp = []
#     for m in range(M):
#         temp.append(h_front[k][m]+h_behind[k][m])  # 拼接
#     order_set.append(temp)

trainX=np.array(___)
trainY = np.array(x)[:,1:]

model = build_model(node_num,max_col,code_length)
# for i in order_set:  # K
#     for j in i:
#         trainX.append(j)

model.fit(trainX,trainY,batch_size=node_num,epochs=10,verbose=1)

