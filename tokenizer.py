import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.model import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhiteSpace
from joblib import Path
from datasets import load_dataset
from torch.utils.data import random_split,Dataset,DataLoader
from dataset import BilingualDataset
from 
def get_all_sentences(ds,lang):
  for item in ds:
    yield item["translation"][lang]
def get_or_build_tokenizer(config,ds,lang):
  tokenizer_path = Path(config["tokenizer_file"].format(lang))
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_tokens="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]","[EOS]","[SOS]","[PAD]")
    tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer
def get_ds(config):
  ds_raw = load_dataset("opus_books",f'{config[lang_src]}-{config[lang_tgt]}',split="train")

  tokenizer_src = get_or_build_tokenizer(confid,ds,config[lang_src])
  tokenizer_tgt = get_or_build_tokenizer(confid,ds,config[lang_tgt])

  train_ds_size = int(0.9*len(ds_raw)
  valid_ds_size = len(ds_raw) - train_ds_size

  train_ds = BilingualDataset(train_ds_size,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
  train_ds = BilingualDataset(valid_ds_size,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])

  max_src_len = []
  max_tgt_len = []
  for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_src_len = max(max_src_len,len(src_ids))
        max_tgt_len = max(max_tgt_len,len(tgt_ids))
  print(f'Max length of source sentence: {max_src_len}')
  print(f'Max length of target sentence: {max_tgt_len}')
             
          
  train_data_loader = DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
  valid_data_loader = DataLoader(valid_ds,batch_size=1,shuffle=True)

  return train_data_loader,valid_data_loader,tokenizer_src,tokenizer_tgt
