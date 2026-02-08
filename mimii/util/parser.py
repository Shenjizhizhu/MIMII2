import os
import argparse
import torch
import numpy as np
from flax import nnx
import tomllib
import math

class readconfig:
    def __init__(self):
        self.parser = self.get_parser_from_toml()
        self.process_parser()
        self.pkl_file_path = self.config["data"]["pkl_file_path"]
        self.batch_size = self.config["data"]["batch_size"]
        self.vocab_size = self.config["data"]["vocab_size"]
        self.padding_size = self.config["data"]["padding_size"]
        self.UNK = self.config["data"]["UNK"]
        self.PAD = self.config["data"]["PAD"]
        self.N = self.config["data"]["N"]
        self.P = self.config["data"]["P"]
        self.dataset = self.config["split"]["dataset"]
        self.test_size = self.config["split"]["test_size"]
        self.random_state = self.config["split"]["random_state"]
        self.model = self.config["model"]["model"]
        self.input_dim = self.config["model"]["input_dim"]
        self.d_model = self.config["model"]["d_model"]
        self.num_classes = self.config["model"]["num_classes"]
        self.epochs = self.config["train"]["epochs"]
        self.loss = self.config["train"]["loss"]
        self.optim = self.config["optimizer"]["optim"]
        self.lr = self.config["optimizer"]["lr"]
    def get_parser_from_toml(self):
        parser = argparse.ArgumentParser(description = 'MIMII参数')
        parser.add_argument('--config',type = str,required = False,default = '/home/shenji/shenji111/MIMIITR/config/config.toml',help = '配置文件路径')
        return parser
    def process_parser(self):
        self.p = self.parser.parse_args()
        if self.p.config.endswith('toml'):
            with open(self.p.config,'rb') as f:
                self.config = tomllib.load(f)
        else:
            raise ValueError("不是.toml格式的文件")
        
class Embedding(nnx.Module):
    def __init__(self,vocab_size):
        super(Embedding,self).__init__()
        self.embedding = nnx.Embedding(vocab_size,self.d_model,padding_idx = self.PAD)

    def forward(self,x):
        for i in range(len(x)):
            x[i].extend([self.UNK] * (self.padding_size - len(x[i])))
        else:
            x[i] = x[i][:self.padding_size]
        x = self.embedding(torch.tensor(x))
        return x

class Positional_Encoding(nnx.Module):
    def __init__(self,d_model):
        super(Positional_Encoding,self).__init__()
        self.d_model = d_model

    def forward(self,seq_len,embedding_dim):
        positional_encoding = np.zeros((seq_len,embedding_dim))
        for pos in range(positional_encoding.sahpe[0]):
            for i in range(positional_encoding.shape[1]):
                positional_encoding[pos][i] = math.sin(pos/(10000**(2*i/self.d_model))) if i % 2 == 0 else math.cos(pos/(10000**(2*i/self.d_model)))
        return torch.from_numpy(positional_encoding)
    