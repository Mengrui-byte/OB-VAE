"""Learned channel and ordered temporal compression for a complete window."""

import math

import torch
from torch import nn

from model.autoencoder.frame_matrix_codec import FrameMatrixCodec


class WindowMatrixCodec(nn.Module):
    def __init__(self,input_dim,window_size=256,rows=16,frame_dim=256,matrix_dim=1024,
                 kind='residual',hidden_dim=512,depth=2,seed=42):
        super().__init__()
        if window_size%rows or rows<1 or window_size<rows:raise ValueError('Window must divide into matrix rows')
        self.window_size,self.rows,self.frame_dim,self.matrix_dim=window_size,rows,frame_dim,matrix_dim
        self.patch_size=window_size//rows
        if matrix_dim%frame_dim or matrix_dim>self.patch_size*frame_dim:raise ValueError('DCT initialization needs an integer number of time modes')
        self.frame=FrameMatrixCodec(input_dim,frame_dim,kind,hidden_dim,seed,depth)
        self.time=FrameMatrixCodec(self.patch_size*frame_dim,matrix_dim,kind,hidden_dim,seed+1,depth)
        self.initialize_time()

    @torch.no_grad()
    def initialize_time(self):
        n=self.patch_size;modes=self.matrix_dim//self.frame_dim
        positions=torch.arange(n,dtype=torch.float32)+.5
        frequencies=torch.arange(modes,dtype=torch.float32)
        basis=torch.cos(math.pi/n*frequencies[:,None]*positions[None])*math.sqrt(2/n)
        basis[0]/=math.sqrt(2)
        basis=torch.kron(basis,torch.eye(self.frame_dim))
        self.time.initialize_pca(torch.zeros(n*self.frame_dim),basis)

    def encode(self,x):
        if x.ndim!=3 or x.shape[1]!=self.window_size:raise ValueError('Wrong full-window length')
        frames=self.frame.encode(x)
        return self.time.encode(frames.reshape(x.shape[0],self.rows,-1))

    def decode(self,matrix):
        if matrix.ndim!=3 or tuple(matrix.shape[1:])!=(self.rows,self.matrix_dim):raise ValueError('Wrong whole-window matrix shape')
        frames=self.time.decode(matrix).reshape(matrix.shape[0],self.window_size,self.frame_dim)
        return self.frame.decode(frames)

    def forward(self,x):
        matrix=self.encode(x)
        return dict(matrix=matrix,reconstruction=self.decode(matrix))
