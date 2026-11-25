import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'base'))
import torch
from torch import nn
from torch.nn import functional as F
from core import trunk,seed_all,supcon
import run as base

class Representation(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(trunk(),nn.Linear(128,48))
        self.classifier=nn.Sequential(nn.Linear(48,64),nn.SiLU(),nn.Linear(64,9))
        self.decoder=nn.Sequential(nn.Linear(48,128),nn.SiLU(),nn.Linear(128,256),nn.SiLU(),nn.Linear(256,1024))
    def encode(self,x):return self.encoder(x)
    def decode(self,z):return self.decoder(z).reshape(-1,2,512)

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__();self.domain=nn.Embedding(2,8)
        self.net=nn.Sequential(nn.Linear(48+9+32+8,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,48))
    def forward(self,z,t,p,d):
        frequency=torch.exp(torch.arange(16,device=z.device)*(-math.log(10000)/15))
        phase=t[:,None].float()*frequency[None]
        return self.net(torch.cat([z,p,phase.sin(),phase.cos(),self.domain(d)],-1))

class DomainWitness(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(trunk(),nn.Linear(128,2))
    def forward(self,x):return self.net(x)

@torch.no_grad()
def encode(model,x):return torch.cat([model.encode(a) for a in x.split(256)])

def train_representation(xs,y,xt,args,out):
    seed_all(args.seed+100);model=Representation().to(xs.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    g=torch.Generator(device=xs.device).manual_seed(args.seed+101);history=[]
    for step in range(1,args.representation_steps+1):
        si=base.balanced(y,g);ti=base.sample(len(xt),len(si),xt.device,g)
        x=torch.cat([xs[si],xt[ti]]);z=model.encode(x);n=len(si)
        reconstructed=model.decode(z)
        ce=F.cross_entropy(model.classifier(z[:n]),y[si]);rec=F.mse_loss(reconstructed,x)
        cycle=F.mse_loss(model.encode(reconstructed),z.detach())
        cl=supcon(F.normalize(z[:n],dim=-1),y[si])
        loss=ce+rec+.1*cl+.1*cycle
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%300==0 or step==args.representation_steps:
            row={'step':step,'ce':float(ce.detach()),'reconstruction':float(rec.detach()),'source_contrastive':float(cl.detach()),'latent_cycle':float(cycle.detach())}
            history.append(row);print({'phase':'representation',**row},flush=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    base.dump(out/'representation_training.json',history);return model

def train_diffusion(u,condition,ns,args,out):
    seed_all(args.seed+200);net=Denoiser().to(u.device)
    opt=torch.optim.AdamW(net.parameters(),lr=.0002)
    g=torch.Generator(device=u.device).manual_seed(args.seed+201)
    ab=torch.cumprod(1-torch.linspace(.0001,.02,200,device=u.device),0);history=[]
    for step in range(1,args.diffusion_steps+1):
        ix=torch.cat([base.sample(ns,32,u.device,g),base.sample(len(u)-ns,32,u.device,g)+ns])
        t=torch.randint(200,(64,),device=u.device,generator=g)
        noise=torch.randn((64,48),device=u.device,generator=g);a=ab[t,None]
        z=a.sqrt()*u[ix]+(1-a).sqrt()*noise;domain=(ix>=ns).long()
        loss=F.mse_loss(net(z,t,condition[ix],domain),noise);assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if step%400==0 or step==args.diffusion_steps:
            row={'step':step,'noise_mse':float(loss.detach())};history.append(row);print({'phase':'diffusion',**row},flush=True)
    net.eval()
    for p in net.parameters():p.requires_grad_(False)
    base.dump(out/'diffusion_training.json',history);return net,ab

def train_domain_witness(xs,xt,args,out):
    seed_all(args.seed+400);net=DomainWitness().to(xs.device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    g=torch.Generator(device=xs.device).manual_seed(args.seed+401)
    labels=torch.cat([torch.zeros(32),torch.ones(32)]).long().to(xs.device)
    for step in range(1,args.witness_steps+1):
        x=torch.cat([xs[base.sample(len(xs),32,xs.device,g)],xt[base.sample(len(xt),32,xt.device,g)]])
        loss=F.cross_entropy(net(x),labels)
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
    net.eval()
    for p in net.parameters():p.requires_grad_(False)
    torch.save({'state_dict':net.state_dict(),'step':args.witness_steps},out/'domain_witness_last.pt');return net
