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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from threading import stack_size
class VQVAE(nn.Module):
    def __init__(self, acousticDim, linguisticDim, hiddenDim1, hiddenDim2, encoderOutDim, lstm2Dim, acousticOutDim, num_class):
        super(VQVAE, self).__init__()

        #self.inputDim = acousticDim + linguisticDim
        self.inputDim = acousticDim

        self.num_class = num_class
        self.z_dim = encoderOutDim
        self.quantized_vectors = nn.Embedding(self.num_class, self.z_dim)
        self.quantized_vectors.weight = nn.init.normal_(self.quantized_vectors.weight, 0, 1)
        
        self.lstm2Dim = lstm2Dim
        self.acousticOutDim = acousticOutDim

        self.calc_weight = nn.Linear(hiddenDim1*2, 1, bias=False).to(device)

        self.avgpool = nn.AvgPool1d(3,ceil_mode = True)
        self.maxpool = nn.MaxPool1d(200,ceil_mode = True)

        #ここからEncoder
        self.fc11 = nn.Linear(self.inputDim, self.inputDim).to(device)
        self.lstm1 = nn.LSTM(input_size = self.inputDim,
                            hidden_size = hiddenDim1,
                            bidirectional = True,
                            batch_first = True).to(device)
        #batch_firstをTrueにすると，(seq_len, batch, input_size)と指定されている入力テンソルの型を(batch, seq_len, input_size)にできる
        self.fc12 = nn.Linear(hiddenDim1*2, encoderOutDim).to(device)
        """
        #ここからDecoder
        #self.fc21 = nn.Linear(encoderOutDim + linguisticDim, lstm2Dim)
        self.lstm2 = nn.LSTM(input_size = self.lstm2Dim,
                            hidden_size = hiddenDim2,
                            bidirectional = True,
                            batch_first = True)
        self.fc22 = nn.Linear(hiddenDim2*2, acousticOutDim)
        
        self.fc21 = nn.Linear(encoderOutDim + linguisticDim, hiddenDim2*2)
        """

        self.LeakyReLU = nn.LeakyReLU(0.1)

    def encode(self, x):
        x_size = len(x)
        output = [0]*x_size
        x11=[0]*x_size
        x11_relu=[0]*x_size
        for i in range(x_size):
            x11[i] = self.fc11(x[i])
            x11_relu[i] = self.LeakyReLU(x11[i])
            output[i], (_, _) = self.lstm1(x11_relu[i].reshape(1, x11_relu[i].size()[0], self.inputDim)) #LSTM層 self.inputDim=x[i].size()[1]

            
            weight = torch.softmax(self.calc_weight(output[i]), 1)
            output[i] = torch.sum(weight * output[i], dim=1)
            output[i] = torch.squeeze(output[i], dim=0)
                     
            """
            output_last = output[i][0][-1]
            output[i] = torch.transpose(output[i], 1, 2)
            #output[i] = self.avgpool(output[i])
            output[i] = self.maxpool(output[i])
            while output[i].size()[2] != 1:
                output[i] = self.maxpool(output[i])
            output[i] = output[i].view(-1)
            
            
            
            #output[i] = torch.cat([output[i],output_last], dim = 0)
            #output[i] = torch.mean(output[i],1).reshape(output[i].size()[2])#mean pooling
            """
        out = torch.cat(output)#out.size():[batch_len*hiddenDim1*2]
        out = out.reshape(x_size, *output[0].shape)#out.size():[batch_len,hiddenDim1*2]
        out2 = self.LeakyReLU(out)
        h1 = self.fc12(out) #全結合層
        return h1

    """
    def quantize_z(self, z_unquantized):
        error = torch.sum((z_unquantized.detach().repeat(1, 1, self.num_class).view(-1, self.num_class, self.z_dim) - self.quantized_vectors.weight.detach().view(self.num_class, self.z_dim))**2, dim=2)
        quantized_z_indices = torch.argmin(error, dim=1)
        quantized_z = self.quantized_vectors.weight[quantized_z_indices].view(-1, z_unquantized.size()[1], self.z_dim)
        z = z_unquantized - z_unquantized.detach() + quantized_z
        #print(self.quantized_vectors.weight)
        #return quantized_z_indices
        return z
    """

    """
    def decode(self, z, linguistic_features):
        z_size = z.size()[1]
        output = [0]*z_size
        for i in range(z_size):
            input = torch.cat((z[0][i].reshape(1,z[0][i].size()[0]).expand(linguistic_features[i].size()[0],-1), linguistic_features[i]),1)
            input2 = self.fc21(input)
            input2 = F.relu(input2)
            input2=input2.reshape(1, input2.size()[0],input2.size()[1])
            #input2, (hidden, cell) = self.lstm2(input2) #LSTM層 self.lstm2Dim=input2[i].size()[1]
            output[i] = self.fc22(input2)
            output[i] = torch.tanh(output[i].reshape(output[i].size()[1], self.acousticOutDim))
            #output[i] = torch.transpose(output[i], 0, 1)

        return output
    """
    def forward(self, acoustic_features, linguistic_features):
        inputs = []
        z_not_quantized = self.encode(acoustic_features)
        #print(z_not_quantized)
        #quantized_z_indices = self.quantize_z(torch.reshape(z_not_quantized, (1, z_not_quantized.size()[0], z_not_quantized.size()[1])))
        #output = self.decode(z, linguistic_features)

        """
        quantized_z_indices_output = []
        for i in range(quantized_z_indices.size()[0]):
            quantized_z_indices_output.append(torch.tensor([quantized_z_indices[i]],dtype=torch.float))
        """

        return z_not_quantized