import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
import os

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence, symbols
import pyopenjtalk
from prepare_tg_accent import pp_symbols
from convert_label import openjtalk2julius

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def paulabel_convert(text,fullcontext_labels):
    fullcontext_labels_after=[]
    paulabel=[]
    for c in text:
        if c=="、":
            paulabel+="1"
        elif c=="・":
            paulabel+="6"
        elif c=="。":
            paulabel+="5"
        count=0
        label_vec=[0,0,0,0,0]
    for line in fullcontext_labels:
        line_split1=re.split(' |/A', line)
        line_split2=re.split('[-|+|=|^]',line_split1[0])
        for i in range(0,4):
            label_vec[i]=label_vec[i+1]
        if line_split2[4]=="pau":
            label_vec[4]=int(paulabel[count])
            count+=1
        else:
            label_vec[4]=0
        for i in range(5):
            if label_vec[i]==0:
                continue
            elif label_vec[i]<=4 or label_vec[i]==6:
                if line_split2[i]=="pau":
                    line_split2[i]="pau"
            elif label_vec[i]==5:
                if line_split2[i]=="pau":
                    line_split2[i]="pauA" #句点の場合
        line_phoneme=line_split2[0]+"^"+line_split2[1]+"-"+line_split2[2]+"+"+line_split2[3]+"="+line_split2[4]
        line_after=line_phoneme+"/A"+line_split1[1]
        fullcontext_labels_after.append(line_after)
    return fullcontext_labels_after

def preprocess_japanese(text:str):
    fullcontext_labels = pyopenjtalk.extract_fullcontext(text)
    fullcontext_labels = paulabel_convert(text,fullcontext_labels)#paulabel用変換
    phonemes , accents = pp_symbols(fullcontext_labels)
    phonemes = [openjtalk2julius(p) for p in phonemes if p != '']
    return phonemes, accents

def phoneme_from_TextGrid(TextGrid):
    with open(TextGrid) as f:
        texts = f.readlines()
        phonemes = []
        for text in texts:
            if text.split() != []:
                if text.split()[0] == "text":
                    #print(text.split()[0])
                    phonemes.append(text.split()[2][1:-1])
    return phonemes[1:-1]




def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    use_accent = preprocess_config["preprocessing"]["accent"]["use_accent"]
    use_kumatu = preprocess_config["preprocessing"]["accent"]["use_kumatu"]
    for batch in batchs:
        batch = to_device(batch, device)
        accents = None
        kumatu = None
        if use_accent:
            if use_kumatu:
                accents = batch[-2]
                kumatu = batch[-1]
                batch = batch[:-2]
                output = model(*(batch[2:]),accents=accents, kumatu = kumatu)
            else:
                accents = batch[-1]
                batch = batch[:-1]
                output = model(*(batch[2:]),accents=accents)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
                accents=accents,
                kumatu = kumatu
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_from_pred_ap_path"],
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--kumatu",
        type=str,
        default=None,
        help="kumatu to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default=None,
        help="speech_name to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speech_No",
        type=str,
        default=None,
        help="speech_No to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
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
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
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
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    accent_to_id = {'0':0, '[':1, ']':2, '#':3}

    kumatu = args.kumatu

    with open(args.source) as f:
        l = f.readlines()
    
    for i in range(len(l)):
        speech_No = l[i].split("|")[0]
        

        if args.mode == "single":
            speakers = np.array([args.speaker_id])
            if preprocess_config["preprocessing"]["text"]["language"] == "en":
                texts = np.array([preprocess_english(args.text, preprocess_config)])
            elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
                texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
            elif preprocess_config["preprocessing"]["text"]["language"] == "ja":
                if speech_No != None:
                    basename = speech_No
                    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "accent",basename+ '.accent')) as f:
                        accents = f.read()
                    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "kumatu_pred",basename+ '.kumatu')) as f:
                        kumatu = f.read()
                    phonemes = phoneme_from_TextGrid(os.path.join(preprocess_config["path"]["preprocessed_path"], "TextGrid",preprocess_config["path"]["preprocessed_path"].split("/")[-1],basename+ '.TextGrid'))
                    texts = np.array([[symbol_to_id[t] for t in phonemes]])
                    accents = np.array([[accent_to_id[a] for a in accents]])
                    kumatu = np.array([[float(a) for a in kumatu.split()]])
                else:
                    phonemes, accents = preprocess_japanese(args.text)
                    print(phonemes,accents)
                    texts = np.array([[symbol_to_id[t] for t in phonemes]])
                    if preprocess_config["preprocessing"]["accent"]["use_accent"]:
                        accents = np.array([[accent_to_id[a] for a in accents]])
                    else:
                        accents = None
                    if preprocess_config["preprocessing"]["accent"]["use_kumatu"]:
                        kumatu = np.array([[float(a) for a in kumatu.split()]])
                    else:
                        kumatu = None
            
            ids = raw_texts = [basename[:100]]

            text_lens = np.array([len(texts[0])])
            print(text_lens)
            batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens),accents,kumatu)]

        control_values = args.pitch_control, args.energy_control, args.duration_control

        output_path = train_config["path"]["result_from_pred_ap_path"]
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
