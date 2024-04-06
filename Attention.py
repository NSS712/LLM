import torch
import torch.nn as nn
import numpy as np

class Attention(nn.modules):
    def __init__(self):
        super(Attention, self).__init()
        
    def forward(query, key, value, mask=None):
        '''
            一个句子"Hello, how are you?"长度是6, embedding维度是300, 那么Q, K, V都是(6, 300)的矩阵。
            shape = (len_k, d_k)
            d_k 是单个token的向量维数
        '''
        d_k = query.shape(-1)
        scores = torch.matmul(query ,key.transpose(-1, -2) / np.sqrt(d_k))
        # scores.shape = (len_k,  len_k)
        # 添加mask， 把True的位置设置为 负无穷
        scores.masked_fill_(mask, -1e9) # softmax之后就会变成无穷小
        attn = nn.softmax(scores)
        context = torch.matmul(attn, value)
        # context.shape = (len_k, d_k)
        return context



class MultiHeadAttention(nn.modules):
    def __init__(self, len_k, n_heads, d_k):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(len_k, d_k * n_heads)
        self.W_K = nn.Linear(len_k, d_k * n_heads)
        self.W_V = nn.Linear(len_k, d_k * n_heads)
        self.linear = nn.Linear(n_heads * d_k, len_k)
        self.layer_norm = nn.LayerNorm(len_k)
    
    def forward():
        
    