import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)





class PositionalEncodings(nn.Module):
    def __init__(self,d_model,seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        pe = torch.zeros(seq_len,d_model)
        position = torch.arange(0,seq_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0,d_model,2).float()/d_model)*(-torch.log(torch.tensor(10000.0))))

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)
    def forward(self,x):
        x = x + self.pe[:,:x.shape[1],:]
        return x,self.pe
