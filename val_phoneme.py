import argparse
import os

import torch
import yaml
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss3
from dataset import Dataset
from text import symbols
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate phone number
phonenumber_to_watch_back=7
phonenumber_to_watch_forward=1
calc_number = 1+phonenumber_to_watch_back+1+phonenumber_to_watch_forward

def get_pau_location(idtexts,pau_id,lengths,kumatu,max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    lenmask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    lenmask=~lenmask
    paumask=torch.zeros(batch_size,max_len).to(device)
    paumask=paumask==1
    for i in range(kumatu.size()[0]):
        for j in range(kumatu.size()[1]):
            if j == kumatu.size()[1]-1:
                if kumatu[i][j]!=0:
                    paumask[i][j] = True
            else:
                if kumatu[i][j] != 0 and kumatu[i][j] != kumatu[i][j+1]:
                    paumask[i][j] = True
    mask_output = paumask & lenmask
    return mask_output

def get_all_masks(pau_loc,back,forward,lengths,max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    lenmask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    lenmask=~lenmask
    falsetensor=torch.zeros(batch_size,1).to(device)
    falsetensor=falsetensor==1

    mask_back=[]
    mask = pau_loc
    for i in range(back):
        mask_split = torch.split(mask,(1,max_len-1),dim=1)
        mask = torch.cat((mask_split[1],falsetensor),1)
        mask = mask & lenmask
        mask_back.append(mask)
    mask_back.reverse()
    mask_forward=[]
    mask = pau_loc
    for i in range(forward):
        mask_split = torch.split(mask,(max_len-1,1),dim=1)
        mask = torch.cat((falsetensor,mask_split[0]),1)
        mask = mask & lenmask
        mask_forward.append(mask)
    maskall_output=mask_back+[pau_loc]+mask_forward
    return maskall_output

def evaluate(model, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # use accent info?
    use_accent = preprocess_config['preprocessing']["accent"]["use_accent"]
    use_kumatu = preprocess_config['preprocessing']["accent"]["use_kumatu"]
    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    #batch_size = preprocess_config["preprocessing"]["val_size"]
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    # Get loss function
    Loss = FastSpeech2Loss3(preprocess_config, model_config).to(device)

    # Symbol to id function
    pause_phone=["sp"]
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    pau_id = np.array([symbol_to_id[t] for t in pause_phone])

    align_kumatu = True#句末ラベルをある値に揃える
    if align_kumatu:
        align_kumatu_num = 3

    # Evaluation
    loss_all = [[np.empty(1) for _ in range(4)] for _ in range(calc_number)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Paumasks
                idtexts=batch[3]
                src_lens=batch[4]
                max_src_lens=batch[5]

                # Forward
                if use_accent:
                    if use_kumatu:
                        accents = batch[-2]
                        kumatu = batch[-1]
                        if align_kumatu:
                            for i in range(kumatu.size()[0]):
                                for j in range(kumatu.size()[1]):
                                    if kumatu[i,j] != 0:
                                        kumatu[i,j] = align_kumatu_num
                        batch = batch[:-2]
                        output = model(*(batch[2:]),accents=accents, kumatu = kumatu)
                    else:
                        accents = batch[-1]
                        batch = batch[:-1]
                        output = model(*(batch[2:]),accents=accents)
                else:
                    output = model(*(batch[2:]))
                
                pau_loc = get_pau_location(idtexts,pau_id,src_lens,kumatu,max_src_lens)

                all_masks = get_all_masks(pau_loc,phonenumber_to_watch_back,phonenumber_to_watch_forward,src_lens,max_src_lens)
                losses_allmasks = Loss(batch, output, all_masks)
                # Cal Loss
                
                for j in range(len(losses_allmasks)):
                    losses=losses_allmasks[j]
                    for i in range(len(losses)):
                        loss_all[j][i] = np.append(loss_all[j][i],losses[i].to('cpu').detach().numpy().copy())

    loss_all_reshape = [[0 for _ in range(calc_number)] for _ in range(4)]
    for o in range(len(loss_all)):
        for p in range(len(loss_all[o])):
            arrays=np.split(loss_all[o][p],[1])
            loss_all_reshape[p][o]=arrays[1]
    return loss_all_reshape


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=200000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    loss_all = evaluate(model, configs)
    title_name=["total","pitch","energy","duration"]
    for i in range(len(loss_all)):
        fig, ax = plt.subplots()
        bp = ax.boxplot(loss_all[i][0:], showmeans=True, sym="")
        #bp = ax.boxplot(loss_all[i][0:], showmeans=True)
        name = [str(n) for n in list(range(-phonenumber_to_watch_back,phonenumber_to_watch_forward+1,1))]
        name.append("all")
        ax.set_xticklabels(name)
        plt.title(title_name[i])
        plt.grid()
        plt.savefig("graph/val_phoneme2/"+str(args.restore_step)+title_name[i]+".png")
    
    for i in range(len(loss_all)):
        for j in range(len(loss_all[i])):
            np.save("graph/val_phoneme2/loss"+str(i)+"_"+str(j),loss_all[i][j])
