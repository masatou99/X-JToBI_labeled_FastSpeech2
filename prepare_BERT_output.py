import argparse
import pathlib
from pathlib import Path
import re
import sys

from tqdm import tqdm

from convert_label import read_lab
from kumatu_classification import VQVAE

from scipy.io import wavfile

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
import math
import glob
import os
from torch.optim import SGD
import pandas as pd
import random

import pyworld as pw

import transformers
from transformers import BertModel
import MeCab
from transformers import BertJapaneseTokenizer

import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def use_BERT(text, model, tknz):
    enc = tknz.encode(text)
    enc = torch.LongTensor(enc).unsqueeze(0)
    BERToutput = model(enc)
    return BERToutput[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocess_config", type=str, help="path to preprocess.yaml")
    parser.add_argument('output',type=str,help='path to output accent and tg file')
    parser.add_argument('speaker',type=str,help='speaker name')
    parser.add_argument('rawdata',type=str,help='path to rawdata. this program searchs for .lab files in specified directory and subdirectories')
    

    args = parser.parse_args()
    lab_files = pathlib.Path(args.rawdata).glob('**/*.lab')

    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)

    # create output directory
    BERToutput_dir = (Path(args.output)/ 'BERToutput')
    if not BERToutput_dir.exists():
        BERToutput_dir.mkdir(parents=True)
    
    model = BertModel.from_pretrained(preprocess_config["preprocessing"]["BERT"]["BertModel"])
    tknz = BertJapaneseTokenizer.from_pretrained(preprocess_config["preprocessing"]["BERT"]["BertJapaneseTokenizer"])

    # iter through lab files
    for lab_file in tqdm(lab_files):
        basename = os.path.splitext(os.path.basename(lab_file))[0]
        with open(lab_file) as f:
            lines = f.readlines()
        BERT_output = use_BERT(lines[0], model, tknz)
        BERToutput_filename = "{}-BERToutput-{}.npy".format(args.speaker, basename)
        np.save(os.path.join(args.output, "BERToutput", BERToutput_filename), BERT_output.to('cpu').detach().numpy().copy())

        

    
