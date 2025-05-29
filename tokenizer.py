import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import random_split,Dataset,DataLoader
from dataset import BilingualDataset
def get_all_sentences(ds,lang):
  for item in ds:
    yield item["translation"][lang]
def get_or_build_tokenizer(config,ds,lang):
  tokenizer_path = Path(config["tokenizer_file"].format(lang))
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]","[EOS]","[SOS]","[PAD]"])
    tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer
def get_ds(config):
  ds_raw = load_dataset("opus_books",f'{config["lang_src"]}-{config["lang_tgt"]}',split="train")

  tokenizer_src = get_or_build_tokenizer(config,ds_raw,config[lang_src])
  tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config[lang_tgt])

  train_ds_size = int(0.9*len(ds_raw))
  valid_ds_size = len(ds_raw) - train_ds_size
  train_ds_raw,valid_ds_raw = random_split(ds_raw,[train_ds_size,valid_ds_size])

  train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
  valid_ds = BilingualDataset(valid_ds_raw,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])

  max_src_len = 0
  max_tgt_len = 0
  for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_src_len = max(max_src_len,len(src_ids))
        max_tgt_len = max(max_tgt_len,len(tgt_ids))
  print(f'Max length of source sentence: {max_src_len}')
  print(f'Max length of target sentence: {max_tgt_len}')
             
          
  train_data_loader = DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
  valid_data_loader = DataLoader(valid_ds,batch_size=1,shuffle=False)

  return train_data_loader,valid_data_loader,tokenizer_src,tokenizer_tgt

def get_model(config,src_vocab_size,tgt_vocab_size):
  model = build_transformer(src_vocab_size,tgt_vocab_size,config["seq_len"],config["seq_len"])
  return model

def train_model():
  if torch.cuda.is_available():
     device = torch.device("cuda")
  else:
     device = torch.device("cpu")
     print("gpu not available")
    
  Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
  
  train_data_loader,valid_data_loader,tokenizer_src,tokenizer_tgt = get_ds(config)


  
  

  

  
  



