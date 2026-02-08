from flax import nnx
import math
import numpy as np
import torch
from util.parser import *
from feeder.dataset import *
import permute
from functools import partial

class CNN(nnx.Module):
    def __init__(self,rngs:nnx.Rngs,return_latent=True,num_classes = 3):
        self.return_latent = return_latent
        self.conv1 = nnx.Conv(1,32,kernel_size = (3,3),padding = 'SAME',rngs = rngs)
        self.conv2 = nnx.Conv(32,64,kernel_size = (3,3),padding = 'SAME',rngs = rngs)
        self.avg_pool = partial(nnx.avg_pool,window_shape = (2,2),strides = (2,2),padding = 'SAME')
        self.map = nnx.Linear(9984,256,rngs = rngs)
        self.embed = nnx.Linear(256,2,rngs = rngs)
        self.logits = nnx.Linear(2,num_classes,rngs = rngs)

    def __call__(self,x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0],-1)
        x = nnx.relu(self.map(x))
        latent = self.embed(x)
        logits = self.logits(latent)
        if self.return_latent:
            return latent,logits
        else:
            return logits
        
class MutiheadAttention(nnx.Module):
    def __init__(self,d_model,rngs,dim_k,dim_v,num_heads):
        super(MutiheadAttention,self).__init__(rngs)
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.num_heads = num_heads
        
        self.q = nnx.Linear(d_model,dim_k)
        self.k = nnx.Linear(d_model,dim_k)
        self.v = nnx.Linear(d_model,dim_v)

        self.o = nnx.Linear(dim_v,d_model)
        self.norm_fact = 1 / math.sqrt(d_model)
     
    def generate_mask(self,dim):
        matirx = np.ones((dim,dim))
        mask = torch.Tensor(np.tril(matirx))

        return mask == 1
    
    def forward(self,x,y,requires_mask=False):
        assert self.dim_k % self.num_heads == 0 and self.dim_v % self.num_heads == 0

        Q = self.q(x).reshape(-1,x.shape[0],x.shape[1],self.dim_k // self.num_heads)
        K = self.k(x).reshape(-1,x.shape[0],x.shape[1],self.dim_k // self.num_heads)
        V = self.v(x).reshape(-1,y.shape[0],y.shape[1],self.dim_v // self.num_heads)

        attention_score = torch.matmul(Q,K,permute(0,1,3,2,)) * self.norm_fact

        if requires_mask:
            mask = self.generate_mask(x.shape[1])
            attention_score.masked_fill(mask,value=float("-inf"))

        output = torch.matmul(attention_score,V).reshape(y.shape[0],y.shape[1],-1)
        
        output = self.o(output)
        return output

class Feed_Forward(nnx.Module):
    def __init__(self,input_dim,hidden_dim = 2048):
        super(Feed_Forward,self).__init__()
        self.L1 = nnx.Linear(input_dim,hidden_dim)
        self.L2 = nnx.Linear(input_dim,hidden_dim)

    def forward(self,x):
        output = nnx.relu()(self.L1(x))
        output = self.L2(output)
        return output

class Add_Norm(nnx.Module):
    def __init__(self):
        self.dropout = nnx.Dropout(0,1)
        super(Add_Norm,self).__init__()
        
    def forward(self,x,sub_layer,**kwargs):
        sub_output = sub_layer(x,**kwargs)
        x = self.dropout(x + sub_output)

        layer_norm = nnx.LayerNorm(x.size()[1:])
        out = layer_norm(x)
        return out
    
class Encoder(nnx.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.positional_encoding = Positional_Encoding(self.d_model)
        self.muti_atten = MutiheadAttention(self.d_model,self.dim_k,self.dim_v,self.n_heads)
        self.add_norm = Add_Norm()

    def forward(self,x):
        x += self.positional_encoding(x.shape[1],self.d_model)
        output = self.add_norm(x,self.muti_atten,y = x)
        output = self.add_norm(output,self.feed_forward)

        return output
    
class Decoder(nnx.Module):
    def __init__(self):
        super(Decoder,self).__init()
        self.positional_encoding = Positional_Encoding(self.d_model)
        self.muti_atten = MutiheadAttention(self.d_model,self.dim_k,self.dim_v,self.n_heads)
        self.feed_forward = Feed_Forward(self.d_model)
        self.add_norm = Add_Norm()

    def forward(self,x,encoder_output):
        x += self.positional_encoding(x.shape[1],self.d_model)
        output = self.add_norm(x,self.muti_atten,y = x,requires_mask = True)
        output = self.add_norm(output,self.muti_atten,y = encoder_output,requires_mask = True)
        output = self.add_norm(output,self.feed_forward)

        return output
    
class Transformer_layer(nnx.Module):
    def __init__(self):
        super(Transformer_layer,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self,x):
        x_input,x_output = x
        encoder_output = self.encoder(x_input)
        decoder_output = self.decoder(x_input,encoder_output)
        return (encoder_output,decoder_output)
    
class Transformer(nnx.Module):
    def __init__(self,N,vocab_size,output_dim):
        super(Transformer,self).__init__()
        self.embedding_input = Embedding(vocab_size = vocab_size)
        self.embedding_output = Embedding(vocav_size = vocab_size)

        self.output_dim = output_dim
        self.linear = nnx.Linear(self.d_model,output_dim)
        self.softmax = nnx.Softmax(dim = 1)
        self.model = nnx.Sequential(*[Transformer_layer() for _ in range(N)])

    def forward(self,x):
        x_input,x_output = x
        x_input = self.embedding_input(x_input)    
        x_output = self.embedding_output(x_output)

        _ , output = self.model((x_input,x_output))
        
        output = self.linear(output)
        output = self.softmax(output)

        return output
