import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model_ap, get_vocoder
from utils.tools import log_ap, synth_one_sample, count_ac_indices
from dataset_ap_prediction import Dataset

import numpy as np
import random
import math
import matplotlib.pylab as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[1] > max_len:
            raise ValueError("not max_len")

        x_padded = np.pad(
            x, ((0,0),(0, max_len - np.shape(x)[1])), mode="constant", constant_values=PAD
        )
        return x_padded
    def getmask(x, max_len):
        x = np.zeros((x.shape[0],x.shape[1]))
        PAD = 1
        if np.shape(x)[1] > max_len:
            raise ValueError("not max_len")
        x_padded = np.pad(
            x, ((0,0),(0, max_len - np.shape(x)[1])), mode="constant", constant_values=PAD
        )
        return x_padded
    if maxlen:
        output = np.concatenate([pad(x, maxlen) for x in inputs], 0)
        mask = np.concatenate([getmask(x, max_len) for x in inputs], 0)
    else:
        max_len = max(np.shape(x)[1] for x in inputs)
        output = np.concatenate([pad(x, max_len) for x in inputs], 0)
        mask = np.concatenate([getmask(x, max_len) for x in inputs], 0)
    return output, mask

def pad_3D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[1] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[2]
        x_padded = np.pad(
            x, ((0,0),(0, max_len - np.shape(x)[1]),(0,0)), mode="constant", constant_values=PAD
        )
        return x_padded[:, :,:s]
    def getmask(x, max_len):
        x = np.zeros((x.shape[0],x.shape[1],x.shape[2]))
        PAD = 1
        if np.shape(x)[1] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[2]
        x_padded = np.pad(
            x, ((0,0),(0, max_len - np.shape(x)[1]),(0,0)), mode="constant", constant_values=PAD
        )
        return x_padded[:, :,0]
    if maxlen:
        output = np.concatenate([pad(x, max_len) for x in inputs], 0)
        mask = np.concatenate([getmask(x, max_len) for x in inputs], 0)
    else:
        max_len = max(np.shape(x)[1] for x in inputs)
        output = np.concatenate([pad(x, max_len) for x in inputs], 0)
        mask = np.concatenate([getmask(x, max_len) for x in inputs], 0)
    return output, mask

def pad_4D(inputs):
    def pad(x, max_1, max_2):
        PAD = 0
        x_padded = np.pad(
            x, ((0,0),(0, max_1 - np.shape(x)[1]),(0,max_2 - np.shape(x)[2])), mode="constant", constant_values=PAD
        )
        return x_padded
    max_1 = 0
    max_2 = 0
    for i in range(len(inputs)):
        if inputs[i].shape[1] > max_1:
            max_1 = inputs[i].shape[1]
        if inputs[i].shape[2] > max_2:
            max_2 = inputs[i].shape[2]
    output = np.concatenate([pad(x, max_1, max_2) for x in inputs], 0)
    return output

def to_device(data, device):
    (
        phone,
        otherlabels,
        labelmask,
        bertoutput,
        bertmask,
        mask,
        t_p_cor,
        ans
    ) = data

    phone = torch.from_numpy(phone).long().to(device)
    mask = torch.from_numpy(mask).long().to(device)
    bertoutput = torch.from_numpy(bertoutput).float().to(device)
    bertmask = torch.from_numpy(bertmask).long().to(device).to(torch.bool)
    otherlabels = torch.from_numpy(otherlabels).long().to(device)
    labelmask = torch.from_numpy(labelmask).long().to(device).to(torch.bool)
    t_p_cor = torch.from_numpy(t_p_cor).float().to(device)
    ans = torch.from_numpy(ans).long().to(device)

    return (
        phone,
        otherlabels,
        labelmask,
        bertoutput,
        bertmask,
        mask,
        t_p_cor,
        ans
    )

def cal_loss(input, target):
    CEloss = nn.CrossEntropyLoss()
    loss = CEloss(input, target)
    return loss

def calc_correct_num(input, target):
    return (torch.argmax(input, dim=1)==target.view(-1)).count_nonzero().item()

def calc_each_correct_num(input, target):
    t_v = target.view(-1)
    num = np.array([0,0,0])
    correct_num = np.array([0,0,0])
    for i in range(input.size()[0]):
        for j in range(3):
            if t_v[i].item() == j:
                num[j] += 1
                if torch.argmax(input, dim=1)[i].item() == j:
                    correct_num[j] += 1
    return correct_num, num


def calc_labels(input, target):
    num_class = 3
    def make_eval_matrix(num_class, array1, array2):
        matrix_list = []
        for i in range(num_class):
            vec_list = []
            for j in range(num_class):
                if i == j:
                    vec_list.append(array1)
                else:
                    vec_list.append(array2)
            matrix_list.append(np.stack(vec_list, axis=0))
        return np.stack(matrix_list, axis=0)#size=(num_class,num_class,len(array1))
    input_matrix = make_eval_matrix(num_class, np.array([1,0,1,0]),np.array([0,1,0,1]))
    target_matrix = make_eval_matrix(num_class, np.array([1,0,0,1]),np.array([0,1,1,0]))
    input_v = input.view(-1)
    target_v = target.view(-1)
    eval_matrix = np.zeros((num_class,4))
    for i in range(input_v.size()[0]):
        eval_matrix += input_matrix[input_v[i]] * target_matrix[target_v[i]]
    return eval_matrix

def calc_eval_index(eval_matrix):
    num_class = eval_matrix.shape[0]
    acc_each = np.zeros(num_class)
    for i in range(num_class):
        acc_each[i] = (eval_matrix[i,0] + eval_matrix[i,1])/np.sum(eval_matrix[i,:])
    rec_each = np.zeros(num_class)
    for i in range(num_class):
        rec_each[i] = eval_matrix[i,0]/(eval_matrix[i,0] + eval_matrix[i,3])
    pre_each = np.zeros(num_class)
    for i in range(num_class):
        pre_each[i] = eval_matrix[i,0]/(eval_matrix[i,0] + eval_matrix[i,2])
    F_each = np.zeros(num_class)
    for i in range(num_class):
        F_each[i] = 2*rec_each[i]*pre_each[i]/(rec_each[i]+pre_each[i])
    each = [acc_each, rec_each, pre_each, F_each]
    macro = [np.mean(acc_each), np.mean(rec_each), np.mean(pre_each), np.mean(F_each)]
    return macro, each


def plot_data(data, image_path, figsize=(5, 4)):
    print("plot results...")
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    fig_names = ['evaluate']
    for i in range(len(data)):
        axes.imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')
        axes.set_xlabel(fig_names[i])
    #plt.suptitle("\n".join(textwrap.wrap(transcript, 130))) # see https://stackoverflow.com/a/55768955
    #make_space_above(axes, topmargin=1)
    plt.savefig(image_path)
    print("All plots saved!: %s" % image_path)
    plt.close()


def evaluate_ap(model, step, configs, speaker, dataset_name, logger=None):
    preprocess_config, model_config, train_config = configs
    # Get dataset
    dataset = Dataset(
        dataset_name, preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=20,
    )


     # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)

    ap_dataset = []
    for batchs in loader:
        for batch in batchs:
            for i in range(batch[16].shape[1]):
                data = []
                data.append(np.concatenate((np.zeros((1,1)),batch[3],np.zeros((1,1))),1))
                data.append(batch[17])
                data.append(batch[18])
                mask = np.zeros((batch[13].shape[0],batch[13].shape[1]+2), dtype='int64')
                for j in range(batch[13].shape[1]):
                    if batch[13][0,j] == i+1:
                        mask[0,j+1] = 1
                data.append(mask)
                data.append(batch[19][np.newaxis, :, :])
                data.append(batch[16][0,i])
                data.append(batch[0])
                ap_dataset.append(data)
    batch_size = 256
    batchs = []
    for i in range(math.ceil(len(ap_dataset)/batch_size)):
        if i != math.ceil(len(ap_dataset)/batch_size)-1:
            batch= ap_dataset[i * batch_size : (i+1) * batch_size]
        else:
            batch= ap_dataset[i * batch_size :]
        
        phone = []
        for k in range(len(batch)):
            phone.append(batch[k][0])
        phone, _ = pad_2D(phone)
        otherlabels = []
        for k in range(len(batch)):
            otherlabels.append(batch[k][1])
        otherlabels, labelmask = pad_3D(otherlabels)
        bertoutput = []
        for k in range(len(batch)):
            bertoutput.append(batch[k][2])
        bertoutput, bertmask = pad_3D(bertoutput)
        mask = []
        for k in range(len(batch)):
            mask.append(batch[k][3])
        mask, _ = pad_2D(mask)
        t_p_cor = []
        for k in range(len(batch)):
            t_p_cor.append(batch[k][4])
        t_p_cor = pad_4D(t_p_cor)
        ans = []
        for k in range(len(batch)):
            ans.append(batch[k][5])
        ans = np.array(ans)
        filename = []
        for k in range(len(batch)):
            filename.append(batch[k][6])
        batch = [phone, otherlabels, labelmask, bertoutput, bertmask, mask, t_p_cor, ans, filename]
        batchs.append(batch)

    # Evaluation
    mic_accuracy = 0
    size_all = []
    loss_all = []
    num_class = 3
    eval_matrix = np.zeros((num_class,4))
    filename_prev = None
    ap_index_pred_path = os.path.join("preprocessed_data", speaker, "kumatu_pred")
    if not os.path.exists(ap_index_pred_path):
        os.makedirs(ap_index_pred_path)
    for batch in batchs:
        filenames = batch[-1]
        batch = batch[:-1]
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(*(batch))
            loss = cal_loss(output, batch[7])
            accuracy = calc_correct_num(output, batch[7])
            #each_accuracy, each_num = calc_each_correct_num(output, batch[7])
            #each_accuracy_all += each_accuracy
            #each_num_all += each_num
            #print(loss, accuracy/batch[7].size()[0])
            mic_accuracy += accuracy
            pred_label = torch.argmax(output, dim = 1)
            eval_matrix += calc_labels(pred_label, batch[7])
            loss_all.append(loss)
            size_all.append(batch[7].size()[0])
        for i in range(len(filenames)):
            filename = filenames[i]
            if filename != filename_prev:
                if filename_prev != None:
                    """
                    for j in range(ap_index_pred.shape[1]):
                        if np.sum(ap_index_pred[0,j]) == 0:
                            if j != 0:
                                ap_index_pred[0,j] = ap_index_pred[0,j-1]
                            else:
                                ap_index_pred[0,j] = first_label
                    """
                    filepath = os.path.join(ap_index_pred_path, filename_prev[0]+".kumatu")
                    text_content = ""
                    for k in range(ap_index_pred.shape[1]):
                        text_content = text_content + " " + str(int(ap_index_pred[0,k].item()))
                    f = open(filepath, 'w', encoding='UTF-8')
                    f.write(text_content[1:])
                    f.close()
                    #np.save(os.path.join(ap_index_pred_path, speaker+"-ap_index_pred-"+filename_prev[0]+".npy"), ap_index_pred)
                mask = batch[2][i,:]
                length = mask.size()[0]-torch.sum(mask.int())
                ap_index_pred = np.zeros((1,length-2))
                first_label = None
            accentphrase_pos = batch[5][i,:]
            ans = pred_label[i]
            start_pos = -1
            end_pos = length-1
            for k in range(1,length-1):
                if accentphrase_pos[k] == 1:
                    start_pos = k
                    break
            for k in range(start_pos,length-1):
                if accentphrase_pos[k] == 0:
                    end_pos = k
                    break
            if start_pos != -1:
                ap_index_pred[0,start_pos-1:end_pos-1] = np.ones((end_pos - start_pos))*(ans.item()+1)
            if first_label == None:
                first_label = (ans.item()+1)
            filename_prev = filename
    if filename_prev != None:
        """
        for j in range(ap_index_pred.shape[1]):
            if np.sum(ap_index_pred[0,j]) == 0:
                if j != 0:
                    ap_index_pred[0,j] = ap_index_pred[0,j-1]
                else:
                    ap_index_pred[0,j] = first_label
        """
        filepath = os.path.join(ap_index_pred_path, filename_prev[0]+".kumatu")
        text_content = ""
        for k in range(ap_index_pred.shape[1]):
            text_content = text_content + " " + str(int(ap_index_pred[0,k].item()))
        f = open(filepath, 'w', encoding='UTF-8')
        f.write(text_content[1:])
        f.close()
    total_loss = 0
    for i in range(len(size_all)):
        total_loss += loss_all[i]*size_all[i]
    total_loss = total_loss/sum(size_all)
    #print(eval_matrix)
    mic_accuracy = mic_accuracy/sum(size_all)
    #each_accuracy_overall = each_accuracy_all/each_num_all
    
    #macro acc
    macro, each = calc_eval_index(eval_matrix)
    message = "Validation Step {}, Total Loss: {:.4f}, mic_acc: {:.4f}, mac_acc: {:.4f}, mac_rec: {:.4f}, mac_pre: {:.4f}, mac_F: {:.4f}".format(
        step,  total_loss, mic_accuracy, macro[0], macro[1], macro[2], macro[3])
    print(each)

    """
    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=total_loss)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )
    """
    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
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
        "--speaker_name",
        type=str,
        default=None,
        help="speech_name to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default=None,
        help="data_name to synthesize, for single-sentence mode only",
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
    model = get_model_ap(args, configs, device, train=False).to(device)

    message = evaluate_ap(model, args.restore_step, configs, args.speaker_name, args.data_name)
    print(message)
