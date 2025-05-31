import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)

class PositionalEncodings(nn.Module):
    def __init__(self,d_model,seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        pe = torch.zeros(seq_len,d_model).float()
        position = torch.arange(0,seq_len).unsqueeze(1).float()
        div_term = torch.exp(((torch.arange(0,d_model,2))/d_model)*(-math.log(10000.0)))

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)
        
    def forward(self,x):
        x = x + self.pe[:,:x.shape[1],:]
        return x

class LayerNormalisation(nn.Module):
    def __init__(self,d_model,eps=10**-9):
        super().__init__()
        self.d_model = d_model
        self.eps=eps
        self.weights = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    def forward(self,x):
        mean = torch.mean(x,dim=-1,keepdim=True)
        std = torch.std(x,dim=-1,keepdim=True)
        return self.weights*((x-mean)/(std+self.eps)) + self.bias

class FeedForwardLayer(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads==0 , "d_model not divisible by num_heads"
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,q,k,v,mask):
        query = self.w_q(q)  # batch_size,seq_len,d_model @ d_model,d_model  --> batch_size,seq_len,d_model
        key = self.w_k(k)  
        value = self.w_v(v)

        #batch_size,seq_len,d_model-->batch_size,seq_len,num_heads,d_k -->
        # batch_size,num_heads,seq_len,d_k
        query = query.view(query.shape[0],query.shape[1],self.num_heads,self.d_k).transpose(1,2) 
        key = key.view(key.shape[0],key.shape[1],self.num_heads,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.num_heads,self.d_k).transpose(1,2)

        #batch_size,num_heads,seq_len,d_k @ batch_size,num_heads,d_k,seq_len-->batch_size,num_heads,seq_len,seq_len
        attention_scores = query@key.transpose(-2,-1) /math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0,float("-inf"))
            
        attention_scores = torch.softmax(attention_scores,dim=-1)
        attention_scores = self.dropout(attention_scores)

        output = attention_scores @ value #batch_size,num_heads,seq_len,seq_len @batch_size,num_heads,seq_len,d_k-->batch_size,num_heads,seq_len,,d_k
        output = output.transpose(1,2)   #batch_size,seq_len,num_heads,d_k
        x = output.contiguous().view(output.shape[0],output.shape[1],self.num_heads*self.d_k)
        return self.w_o(x)

class ResidualConnections(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.norm = LayerNormalisation(d_model)
    def forward(self,x,sublayer):
        x = self.norm(x+ sublayer(x))
        return x

class ProjectionLayer(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.proj_layer = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return self.proj_layer(x)


class EncoderBlock(nn.Module):
    def __init__(self,d_model,attention,feed_forward):
        super().__init__()
        self.attention =attention
        self.feed_forward = feed_forward
        self.residual_connection1 = ResidualConnections(d_model)
        self.residual_connection2 = ResidualConnections(d_model)
    def forward(self,x,src_mask):
        x = self.residual_connection1(x,lambda x:self.attention(x,x,x,src_mask))
        x = self.residual_connection2(x,lambda x:self.feed_forward(x))
        return x
class Encoder(nn.Module):
    def __init__(self,d_model,layers:nn.ModuleList()):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation(d_model)
    def forward(self,x,src_mask):
        for layer in self.layers:
            x = layer(x,src_mask)
        return self.norm(x)
        

class DecoderBlock(nn.Module):
    def __init__(self,d_model,self_attention,cross_attention,feed_forward):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection1 = ResidualConnections(d_model)
        self.residual_connection2 = ResidualConnections(d_model)
        self.residual_connection3 = ResidualConnections(d_model)
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.residual_connection1(x,lambda x:self.self_attention(x,x,x,tgt_mask))
        x = self.residual_connection2(x,lambda x:self.cross_attention(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connection3(x,lambda x:self.feed_forward(x))
        return x
class Decoder(nn.Module):
    def __init__(self,d_model,layers:nn.ModuleList()):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation(d_model)
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self,
                 encoder:Encoder,
                 decoder:Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed:InputEmbeddings,
                 src_pos:PositionalEncodings,
                 tgt_pos:PositionalEncodings,
                 proj_layer : ProjectionLayer
                ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer
        
    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
        
    def decode(self,tgt,encoder_output,src_mask,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        return self.proj_layer(x)
    

def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int = 512,
                      N: int = 6,
                      num_heads: int = 8,
                      dropout: float = 0.1,
                      d_ff: int = 2048):
    src_embed = InputEmbeddings(d_model,src_vocab_size)
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)

    src_pos = PositionalEncodings(d_model,src_seq_len)
    tgt_pos = PositionalEncodings(d_model,tgt_seq_len)

    encoder_blocks=[]
    for i in range(N):
        encoder_attention = MultiHeadAttention(d_model,num_heads,dropout)
        encoder_feed_forward = FeedForwardLayer(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(d_model,encoder_attention,encoder_feed_forward)
        encoder_blocks.append(encoder_block)

    decoder_blocks=[]
    for i in range(N):
        decoder_self_attention = MultiHeadAttention(d_model,num_heads,dropout)
        decoder_cross_attention = MultiHeadAttention(d_model,num_heads,dropout)
        decoder_feed_forward = FeedForwardLayer(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(d_model,decoder_self_attention,decoder_cross_attention,decoder_feed_forward)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(d_model,nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model,nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)

    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return transformer
