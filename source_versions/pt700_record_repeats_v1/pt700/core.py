import math,random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
LEVELS=[("clean",None),("20dB",20),("10dB",10)]
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def features(w):
    w = w-w.mean(-1, keepdim=True)
    w = w/w.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    return torch.log1p(torch.fft.rfft(w, dim=-1).abs()[..., 1:]/math.sqrt(w.shape[-1]))

def noisy(w, snr, gen):
    w = w-w.mean(-1, keepdim=True)
    n = torch.randn(w.shape, device=w.device, generator=gen)
    n = n/n.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    return w+n*w.square().mean((1, 2), keepdim=True).sqrt()*10**(-snr/20)

def trunk():
    return nn.Sequential(nn.Conv1d(1,32,7,2,3),nn.GroupNorm(4,32),nn.SiLU(),
        nn.Conv1d(32,64,7,2,3),nn.GroupNorm(8,64),nn.SiLU(),
        nn.Conv1d(64,64,5,2,2),nn.GroupNorm(8,64),nn.SiLU(),
        nn.AdaptiveAvgPool1d(16),nn.Flatten(),nn.Linear(1024,128),nn.SiLU())

class Classifier(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=trunk(); self.head=nn.Linear(128,4)
    def forward(self,x): return self.head(self.encoder(x))

class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale): ctx.scale=scale; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.scale*g,None

def reverse(x,scale=1.): return Reverse.apply(x,scale)

def head(n,k): return nn.Sequential(nn.Linear(n,64),nn.SiLU(),nn.Linear(64,k))

def supcon(z,labels,temperature=.1):
    z=F.normalize(z,dim=-1); eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
    scores=z@z.T/temperature; scores=scores.masked_fill(eye,-1e4)
    logp=scores-torch.logsumexp(scores,dim=1,keepdim=True)
    pos=(labels[:,None]==labels[None,:])&~eye
    valid=pos.sum(1)>0
    if not valid.any(): return z.sum()*0
    return -((logp*pos).sum(1)/pos.sum(1).clamp_min(1))[valid].mean()
