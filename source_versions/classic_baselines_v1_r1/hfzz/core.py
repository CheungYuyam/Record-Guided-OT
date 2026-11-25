"""Explicit factor representations and conditional style diffusion.

Local class-preserving directions are defined through decoder -> fault encoder,
not by assuming that architectural style coordinates are physical factors.
"""
import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ['source_only', 'dann', 'reconstruction', 'swap_no_cluster', 'swap_full',
        'isotropic_latent_diffusion', 'foliated_latent_diffusion', 'random_latent_diffusion']
LEVELS = [('clean', None), ('10dB', 10), ('0dB', 0), ('-5dB', -5)]

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
    return nn.Sequential(nn.Conv1d(2,32,7,2,3),nn.GroupNorm(4,32),nn.SiLU(),
        nn.Conv1d(32,64,7,2,3),nn.GroupNorm(8,64),nn.SiLU(),
        nn.Conv1d(64,64,5,2,2),nn.GroupNorm(8,64),nn.SiLU(),
        nn.AdaptiveAvgPool1d(16),nn.Flatten(),nn.Linear(1024,128),nn.SiLU())

class Classifier(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=trunk(); self.head=nn.Linear(128,9)
    def forward(self,x): return self.head(self.encoder(x))

class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale): ctx.scale=scale; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.scale*g,None

def reverse(x,scale=1.): return Reverse.apply(x,scale)
def head(n,k): return nn.Sequential(nn.Linear(n,64),nn.SiLU(),nn.Linear(64,k))

class FactorModel(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=trunk()
        self.fault=nn.Linear(128,32); self.style=nn.Linear(128,16)
        self.fault_head=nn.Linear(32,9)
        self.style_domain=head(16,2); self.fault_domain=head(32,2); self.style_fault=head(16,9)
        self.decoder=nn.Sequential(nn.Linear(48,128),nn.SiLU(),nn.Linear(128,256),nn.SiLU(),nn.Linear(256,1024))
    def encode(self,x):
        h=self.encoder(x)
        return F.normalize(self.fault(h),dim=-1), torch.tanh(self.style(h))
    def decode(self,f,s): return self.decoder(torch.cat([f,s],-1)).reshape(-1,2,512)
    def logits(self,f): return 10*self.fault_head(f)
    def forward(self,x):
        f,s=self.encode(x); return self.logits(f),f,s

def supcon(z,labels,temperature=.1):
    z=F.normalize(z,dim=-1); eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
    scores=z@z.T/temperature; scores=scores.masked_fill(eye,-1e4)
    logp=scores-torch.logsumexp(scores,dim=1,keepdim=True)
    pos=(labels[:,None]==labels[None,:])&~eye
    valid=pos.sum(1)>0
    if not valid.any(): return z.sum()*0
    return -((logp*pos).sum(1)/pos.sum(1).clamp_min(1))[valid].mean()

def cross_covariance(f,s):
    f=(f-f.mean(0))/(f.std(0,unbiased=False)+1e-3)
    s=(s-s.mean(0))/(s.std(0,unbiased=False)+1e-3)
    return (f.T@s/len(f)).square().mean()

@torch.enable_grad()
def local_geometry(model,f,u,mean,std):
    """Jacobian of re-encoded decoded class logits w.r.t. standardized style."""
    u=u.detach().requires_grad_(True); f=f.detach()
    decoded=model.decode(f,u*std+mean)
    logits=model.logits(model.encode(decoded)[0])
    jac=torch.stack([torch.autograd.grad(logits[:,k].sum(),u,retain_graph=k<8)[0] for k in range(9)],1).detach()
    p=logits.detach().softmax(-1).double(); j=jac.double()
    j=j-(j*p[:,:,None]).sum(1,keepdim=True)
    fisher=j.transpose(1,2)@(p[:,:,None]*j)
    ev,v=torch.linalg.eigh((fisher+fisher.transpose(1,2))/2)
    ev=ev[:,-8:].flip(-1).clamp_min(0)
    valid=ev>torch.maximum(ev[:,:1]*1e-6,torch.full_like(ev[:,:1],1e-12))
    q=v[:,:,-8:].flip(-1).transpose(1,2)*valid[:,:,None]
    return {'q':q.float(),'rank':valid.sum(-1),'eigenvalues':ev.float(),'jac':jac}

def chart(geo,mode,rotation):
    q=geo['q']; rank=geo['rank']
    if mode=='random': q=q@rotation
    trans=.25 if mode!='isotropic' else 1.
    base=16/(16+(trans-1)*rank)
    return {'q':q,'base':base,'trans':base*trans}

def covariance(x,c,power=.5):
    q=c['q']; coords=torch.einsum('bkd,bd->bk',q,x)
    projection=torch.einsum('bkd,bk->bd',q,coords)
    return x*c['base'].pow(power)[:,None]+projection*(c['trans'].pow(power)-c['base'].pow(power))[:,None]

def descriptor(c):
    p=c['q'].transpose(1,2)@c['q']
    return ((c['trans']-c['base'])[:,None,None]*p+torch.eye(16,device=p.device)[None]*c['base'][:,None,None]).flatten(1)

class StyleDenoiser(nn.Module):
    def __init__(self):
        super().__init__(); self.domain=nn.Embedding(2,8)
        self.net=nn.Sequential(nn.Linear(16+32+32+8+256,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,16))
    def forward(self,u,t,f,d,c):
        freq=torch.exp(torch.arange(16,device=u.device)*(-math.log(10000)/15))
        phase=t.float()[:,None]*freq[None]
        return self.net(torch.cat([u,f,phase.sin(),phase.cos(),self.domain(d),descriptor(c)],-1))

def schedule(device): return torch.cumprod(1-torch.linspace(1e-4,.02,200,device=device),0)

def forward_noise(u,t,white,c,ab):
    a=ab[t,None]; return a.sqrt()*u+(1-a).sqrt()*covariance(white,c)

def reverse_step(u,white,t,previous,c,ab,clip=False):
    noise=covariance(white,c); u0=(u-(1-ab[t]).sqrt()*noise)/ab[t].sqrt()
    if clip: u0=u0.clamp(-4,4)
    if previous is None:return u0
    return ab[previous].sqrt()*u0+(1-ab[previous]).sqrt()*noise

@torch.no_grad()
def edit(net,u,f,c,white,ab,start=80,steps=16):
    state=forward_noise(u,torch.full((len(u),),start,device=u.device,dtype=torch.long),white,c,ab)
    times=torch.linspace(start,0,steps,device=u.device).long().unique_consecutive()
    target=torch.ones(len(u),device=u.device,dtype=torch.long)
    for i,t in enumerate(times):
        eps=net(state,t.expand(len(u)),f,target,c)
        state=reverse_step(state,eps,t,None if i==len(times)-1 else times[i+1],c,ab,clip=True)
    return state
