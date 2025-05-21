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


class LayerNormalisation(nn.Module):
    def __init__(self,d_model:int,eps=10**-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self,x):
        mean = torch.mean(x,dim=-1,keepdim=True)
        std = torch.std(x,dim=-1,keepdim=True)
        return self.weight * (x-mean)/(std+self.eps) + self.bias


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)  # d_model,d_model
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)  # batch_size,seq_len,d_model @ d_model,d_model --> batch_size,seq_len,d_model
        key = self.w_k(k)
        value = self.w_v(v)

        # batch_size,seq_len,d_model  --> batch_size,seq_len,num_heads,d_k --> batch_size,num_heads,seq_len,d_k
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # batch_size,num_heads,seq_len,d_k @ batch_size,num_heads,d_k,seq_len --> batch_size,num_heads,seq_len,seq_len
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_scores = torch.softmax(attention_scores, dim=-1)

        attention_scores = self.dropout(attention_scores)

        # batch_size,num_heads,seq_len,seq_len @ batch_size,num_heads,seq_len,d_k --> batch_size,num_heads,seq_len,d_k
        output = attention_scores @ value

        # batch_size,num_heads,seq_len,d_k --> batch_size,seq_len,num_heads,d_k --> batch_size,seq_len,d_model
        x = output.transpose(1, 2)
        x = x.contiguous().view(x.shape[0], x.shape[1], self.num_heads * self.d_k)

        return self.w_o(x), output, attention_scores

class ResidualConnection(nn.Module):
    def __init__(self,d_model,dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation(d_model)
    def forward(self,x,sublayer):
        x = x + self.dropout(sublayer(self.norm(x))) 
        return x
