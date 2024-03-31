import torch
import math

class Positional_Embedding(torch.nn.Module):
    '''
        为序列生成位置编码
    '''
    def __init__(self, dk, dropout=0.1, max_length=5000):
        super(Positional_Embedding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_length, dk)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)  #shape 是 [5000, 1]
        # a的b次方，  a^b = e^(blna) torch.exp(b * math.log(b))
        div_term = torch.exp(-math.log(10000) * (torch.arange(0, dk, 2, dtype=float) / dk ))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x  = x + self.pe[:x.size(0), :]
        return x.dropout(x)


if __name__=="__main__":
    pe = Positional_Embedding(512)