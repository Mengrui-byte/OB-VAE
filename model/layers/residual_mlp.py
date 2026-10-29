"""Pre-normalized residual feed-forward block with identity initialization."""

from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self,width,expansion=2):
        super().__init__()
        if width<1 or expansion<1:raise ValueError('Invalid residual MLP width')
        self.norm=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,width*expansion),nn.GELU(),nn.Linear(width*expansion,width))
        nn.init.zeros_(self.ffn[-1].weight);nn.init.zeros_(self.ffn[-1].bias)

    def forward(self,x):return x+self.ffn(self.norm(x))
