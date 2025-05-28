from torch.utils.data import Dataset
import torch.nn as nn
import torch

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,lang_src,lang_tgt,seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")],dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")],dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self,index):
        text_pair = self.ds[index]
        src_text = text_pair["translation"][self.lang_src]
        tgt_text = text_pair["translation"][self.lang_tgt]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 #(for sos and eos)
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 #(for sos)

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens <0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*enc_num_padding_tokens)
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            torch.tensor([self.pad_token]*dec_num_padding_tokens)
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*dec_num_padding_tokens)
        ])

        assert encoder_input.shape[0] == self.seq_len , "error"
        assert decoder_input.shape[0] == self.seq_len , "error"
        assert label.shape[0] == self.seq_len , "error"

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1,                                                                                                          seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)--> 1,seq_len,seq_len
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }  

def causal_mask(size):
    mask=torch.triu(torch.ones((1, size,size)), diagonal=1).type(torch.int)
    return mask==0
