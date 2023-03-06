import argparse
import pathlib
from pathlib import Path
import re
import sys
import numpy as np
from tqdm import tqdm
import os

from convert_label import read_lab


# full context label to accent label from ttslearn
def extract_label(alp, numb, line):
    alp_all = ["p", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    text = line.split()[-1]
    for i in range(len(alp_all)):
        if alp_all[i] == alp:
            text = text.split("/")[i]
            break
    if alp == "a" and numb == 1:
        output = re.split('[+]', text)[0]
    else:
        output = re.split('[+ _ = | @ # ^ % ! & -]', text)[numb-1]
    output = output.split(":")[-1]
    return output

def str_to_int(str, lower, upper):
    if str.isdecimal() == True:
        output = int(str)
        if output > upper:
            output = upper
        elif output < lower:
            output = lower
    else:
        output = -1
    return output

def negative_to_zero(num):
    if num < 0:
        output = 0
    else:
        output = num
    return output


def read_labels(lines):
    variable_num = 14
    output = np.zeros((len(lines),variable_num))
    for i in range(len(lines)):
        #pos:01~24 or xx
        pos_prev =  negative_to_zero(str_to_int(extract_label("b", 1, lines[i]), 1, 24))
        pos =  negative_to_zero(str_to_int(extract_label("c", 1, lines[i]), 1, 24))
        pos_next =  negative_to_zero(str_to_int(extract_label("d", 1, lines[i]), 1, 24))
        #mora:1~49 or xx→1~6 or more
        mora_prev =  negative_to_zero(str_to_int(extract_label("e", 1, lines[i]), 1, 7))
        mora =  negative_to_zero(str_to_int(extract_label("f", 1, lines[i]), 1, 7))
        mora_next =  negative_to_zero(str_to_int(extract_label("g", 1, lines[i]), 1, 7))
        #quest:0 or 1 or xx
        quest_prev =  negative_to_zero(str_to_int(extract_label("e", 3, lines[i]), 0, 1)+1)
        quest =  negative_to_zero(str_to_int(extract_label("f", 3, lines[i]), 0, 1)+1)
        quest_next =  negative_to_zero(str_to_int(extract_label("g", 3, lines[i]), 0, 1)+1)
        #pause:0 or 1 or xx
        pause_prev =  negative_to_zero(str_to_int(extract_label("e", 5, lines[i]), 0, 1)+1)
        pause_next =  negative_to_zero(str_to_int(extract_label("g", 5, lines[i]), 0, 1)+1)
        #accent_phrase_num:1~49 or xx→1~6 or more
        accent_phrase_num =  negative_to_zero(str_to_int(extract_label("i", 1, lines[i]), 1, 7))
        accent_phrase_pos1 =  negative_to_zero(str_to_int(extract_label("f", 5, lines[i]), 1, 7))
        accent_phrase_pos2 =  negative_to_zero(str_to_int(extract_label("f", 6, lines[i]), 1, 7))

        output[i, :] = [pos_prev, pos, pos_next, mora_prev, mora, mora_next, quest_prev, quest, quest_next, pause_prev, pause_next, accent_phrase_num, accent_phrase_pos1, accent_phrase_pos2]
    return output



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lab',type=str,help='path to lab files. this program searchs for .lab files in specified directory and subdirectories')
    parser.add_argument('output',type=str,help='path to output accent and tg file')
    parser.add_argument('speaker',type=str,help='speaker name')

    args = parser.parse_args()
    lab_files = pathlib.Path(args.lab).glob('**/*.lab')

    # create output directory
    ol_dir = (Path(args.output)/ 'otherlabels')
    if not ol_dir.exists():
        ol_dir.mkdir(parents=True)

    # iter through lab files
    for lab_file in tqdm(lab_files):
        basename = os.path.splitext(os.path.basename(lab_file))[0]
        with open(lab_file) as f:
            lines = f.readlines()
        labels = read_labels(lines)
        otherlabels_filename = "{}-otherlabels-{}.npy".format(args.speaker, basename)
        np.save(os.path.join(args.output, "otherlabels", otherlabels_filename), labels)


        

    
