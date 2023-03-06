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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# full context label to accent label from ttslearn
def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))
def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True
def pp_symbols_kumatu(labels, drop_unvoiced_vowels=True):
    kumatu = []
    N = len(labels)
    time = []
    start = 1
    m = 1

    for n in range(len(labels)):
        lab_curr = labels[n]
        if is_num(lab_curr.split()[0])==False:
            continue
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)
        if p3 == "sil":
            continue
        lab_next = labels[n + 1]
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a1_next = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_next)
        if a1 == -50:
            kumatu.append(0)
            continue
        if start == 1:
            time.append([float(lab_curr.split()[0]),None])
            start = 0
        if a1_next == a1 or a1_next == a1+1:
            kumatu.append(m)
            continue
        else:
            kumatu.append(m)
            m += 1
            start = 1
            time[-1][1] = float(lab_curr.split()[1])
    return kumatu, time

def interp0(array):
    return pd.Series(np.where(array==0,np.nan,array)).interpolate().values, np.where(array!=0,1,array)

def smooth(f0):
    num=5#移動平均の個数
    b=np.ones(num)/num
    y=np.convolve(f0, b, mode='same')#移動平均
    y[0] = f0[0]
    y[1] = f0[1]
    y[f0.shape[0]-2] = f0[f0.shape[0]-2]
    y[f0.shape[0]-1] = f0[f0.shape[0]-1]
    return y

def normalize_ndarray(a):
    return (a - a.mean()) / a.std(ddof=1)

def statistics(a):
    out = [[0]]
    for i in range(1,a.shape[0]):
        if a[i] < a[i-1]:
            out.append([-1])
        else:
            out.append([1])
    return np.array(out)

def position(a):
    return np.arange(a.shape[0])/a.shape[0]

def resample(a):
    out = []
    for i in range(math.floor(a.shape[0]/240)):
        out.append(a[240 * i])
    return np.array(out)

def pred_kumatu(wav_file, kumatu_time, model):
    fs, data = wavfile.read(wav_file)
    data = data.astype(np.float64)
    _f0, time = pw.dio(data, fs)    # 基本周波数の抽出
    f0_all = pw.stonemask(data, _f0, time, fs)  # 基本周波数の修正
    """
    f02 = f0[:, None]
    lf0 = f02.copy()
    nonzero_indices = np.nonzero(f02)
    lf0[nonzero_indices] = np.log(f02[nonzero_indices])
    vuv = (lf0 != 0).astype(np.float32)
    lf0 = interp1d(lf0, kind="slinear")
    """
    
    f0_all, _= interp0(f0_all)
    #f0 = resample(f0)
    f0_all = smooth(f0_all)

    output = []
    data_ac = []
    data_lin = []

    for i in range(len(kumatu_time)):
        paubool = time > kumatu_time[i][0]/10000000
        paubool2 = time < kumatu_time[i][1]/10000000
        paubool = paubool & paubool2
        f0=f0_all[paubool]
        f0 = normalize_ndarray(f0)

        f0_len = f0.shape[0]

        sta = statistics(f0)
        pos = position(f0)
        pos2 = pos ** 2

        data_lin.append(torch.tensor([[1,1,1,1] for i in range(f0_len)]).to(device))

        data_ac.append(torch.tensor(np.concatenate([f0.reshape([f0_len, 1]), sta, pos2.reshape([f0_len, 1])], 1).tolist()).to(device))

    out = model(data_ac, data_lin)
    output = (-torch.argmax(out, dim=1) - 1).tolist()
    return output





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lab',type=str,help='path to lab files. this program searchs for .lab files in specified directory and subdirectories')
    parser.add_argument('output',type=str,help='path to output accent and tg file')
    parser.add_argument('speaker',type=str,help='speaker name')
    parser.add_argument('wav_path',type=str,help='path to raw wav file')

    args = parser.parse_args()
    lab_files = pathlib.Path(args.lab).glob('**/*.lab')

    # create output directory
    kumatu_dir = (Path(args.output)/ 'kumatu')
    kumatu_ans_dir = (Path(args.output)/ 'ap_index')
    if not kumatu_dir.exists():
        kumatu_dir.mkdir(parents=True)
    if not kumatu_ans_dir.exists():
        kumatu_ans_dir.mkdir(parents=True)
    
    model = VQVAE(acousticDim=3, linguisticDim=4, hiddenDim1=10, hiddenDim2=20, encoderOutDim=3, lstm2Dim=2, acousticOutDim=1, num_class=3)
    model.load_state_dict(torch.load('/home/sarulab/masaki_sato/デスクトップ/speech/xjtobiLabel/FastSpeech2-JSUT-master/model/long.pth'))

    # iter through lab files
    for lab_file in tqdm(lab_files):
        wav_file = args.wav_path + "/" + os.path.split(lab_file)[1][:-4] + ".wav"
        kumatu = []
        with open(lab_file) as f:
            lines = f.readlines()
        kumatu, time = pp_symbols_kumatu(lines)

        kumatu_label = pred_kumatu(wav_file, time, model)
        for i in range(len(time)):
            kumatu = [kumatu_label[i] if k == i+1 else k for k in kumatu]
        
        kumatu = [str(-i) for i in kumatu]
        kumatu_ans = -np.array(kumatu_label)-1
        np.save(os.path.join(kumatu_ans_dir,args.speaker+"-ap_index-"+os.path.splitext(os.path.basename(lab_file))[0]), kumatu_ans)
        with open(kumatu_dir/ lab_file.with_suffix('.kumatu').name,mode='w') as f:
            f.writelines([' '.join(kumatu)])

        

    
