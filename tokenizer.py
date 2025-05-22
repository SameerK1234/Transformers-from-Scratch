import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.model import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhiteSpace
from joblib import Path
from datasets import load_dataset
from torch.utils.data import random_split
def get_all_sentences(ds,lang):
  for item in ds:
    yield item["translation"][lang]
def get_or_build_tokenizer(config,ds,lang):
  tokenizer_path = Path(config["tokenizer_file"].format(lang))
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_tokens="[UNK]"))
    tokenizer.pre_tokenizer = WhiteSpace()
    trainer = WordLevelTrainer(special_token=["[UNK]","[EOS]","[SOS]","[PAD]")
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

  train_ds , valid_ds = random_split(ds_raw,[train_ds_size,valid_ds_size])

          

  

  
    

