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


class LayerNormalization(nn.Module):
    def __init__(self,d_model:int,eps=10**-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self,x):
        mean = torch.mean(x,dim=-1,keepdim=True)
        std = torch.std(x,dim=-1,keepdim=True)
        return self.weight * (x-mean)/(std+self.eps) + self.bias


class FeedForwardBlock(nn.Module):
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


class MultiHeadAttentionBlock(nn.Module):
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
        self.norm = LayerNormalization(d_model)
    def forward(self,x,sublayer):
        x = x + self.dropout(sublayer(self.norm(x))) 
        return x

class EncoderBlock(nn.Module):

    def __init__(self, d_model, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections1 = ResidualConnection(d_model, dropout)
        self.residual_connections2 = ResidualConnection(d_model, dropout)

    def forward(self, x, src_mask):
        x = self.residual_connections1(x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections2(x, lambda x: self.feed_forward_block(x))
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self,d_model,
                   self_attention_block: MultiHeadAttentionBlock,
                   cross_attention_block: MultiHeadAttentionBlock,
                   feed_forward_block: FeedForwardBlock,
                   dropout):
        super().__init__()
        self.d_model = d_model
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residualconnection1 = ResidualConnection(d_model,dropout)
        self.residualconnection2 = ResidualConnection(d_model,dropout)
        self.residualconnection3 = ResidualConnection(d_model,dropout)
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.residualconnection1(x,lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x = self.residualconnection2(x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residualconnection3(x,lambda x:self.feed_forward_block(x))
        return x
class Decoder(nn.Module):
    def __init__(self,d_model,layers:nn.ModuleList):
        super().__init__()
        self.d_model = d_model
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
                       
class ProjectionLayer(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.proj_layer = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return self.proj_layer(x)     

class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,
                 decoder:Decoder,
                 src_embed:InputEmbeddings,
                 tgt_embed:InputEmbeddings,
                 src_pos:PositionalEncodings,
                 tgt_pos:PositionalEncodings,
                 projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.pos_embed(src)
        return self.encoder(src,src_mask)

    def decode(self,tgt,encoder_output,src_mask,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_ouput,src_mask,tgt_mask)
    def project(self,x):
        x = self.projection_layer(x)
        return x


def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int, 
                      d_model: int=512, 
                      N: int=6, 
                      num_heads: int=8, 
                      dropout: float=0.1, 
                      d_ff: int=2048):
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncodings(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncodings(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
