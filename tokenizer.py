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
from torch.utils.tensorboard import SummaryWriter

import warnings 
warnings.filterwarnings("ignore")
def get_all_sentences(ds,lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config,ds,lang):
    print("get_or_build_tokenizer function ran")
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"])
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    print("get ds ran")
    ds_raw = load_dataset(config['datasource'],f"{config['lang_src']}-{config['lang_tgt']}",split="train")

    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config["lang_tgt"])

    train_ds_size = int(0.9*len(ds_raw))
    valid_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, valid_ds_raw = random_split(ds_raw,[train_ds_size,valid_ds_size])

    train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
    valid_ds = BilingualDataset(valid_ds_raw,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0
    for word in ds_raw:
        src_ids = tokenizer_src.encode(word["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(word["translation"][config["lang_tgt"]]).ids
        max_len_src=max(max_len_src,len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))
    print(f"Max length of src sentences {max_len_src}")
    print(f"Max length of tgt sentences {max_len_tgt}")
    
    train_data_loader = DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
    valid_data_loader = DataLoader(valid_ds,batch_size=1,shuffle=False)

    return train_data_loader,valid_data_loader,tokenizer_src,tokenizer_tgt


def get_model(config,src_vocab_size,tgt_vocab_size):
    print("get model function ran")
    model = build_transformer(src_vocab_size,tgt_vocab_size,config["seq_len"],config["seq_len"])
    return model

    

def train_model(config):
    print("train_model function ran")
    if torch.cuda.is_available():
        print("gpu available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_data_loader,valid_data_loader,tokenizer_src,tokenizer_tgt = get_ds(config)

    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"],eps=10**-9)

    initial_epoch =0
    global_step =0 
    preload = config["preload"]
    
    if preload =="latest":
        model_filename=latest_weights_file_path(config)
    elif preload:
        model_filename=get_weights_file_path(config, epoch)
    else:
        model_filename=None

    if model_filename:
        print(f"Preloading Start {model_filename}")
        state = model.load(model_filename)
        model.load_state_dict = state["model_state_dict"]
        optimizer.load_state_dict = state["optimizer_state_dict"]
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
    else:
        print("no existing model found starting from scratch")

    loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]")).to(device)

    for epoch in range(initial_epoch,config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_data_loader,desc=f"Processing epoch {epoch}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.encode(encoder_input,encoder_mask)
            decoder_output = model.decode(decoder_input,encoder_output,encoder_mask,decoder_mask)
            proj_output = model.project(decoder_output)
            label = batch["label"].to(device)  #(B * seq_len)

            loss = loss_function(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))

            batch_iterator.set_postfix(loss=loss.item())

            writer.add_scalar("train_loss",loss.item(),global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step+=1
    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save({
        "epoch":epoch,
        "model_state_dict":model.state_dict,
        "optimizer_state_dict":optimizer.state_dict,
        "global_step":global_step
    },model_filename)

    if __name__=="__main__":
        config = get_config()
        train_model(config)
