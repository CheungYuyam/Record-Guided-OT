import torch
from torch import nn
from torch.nn import functional as F
from core import trunk,seed_all,supcon
import run as base
class Representation(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(trunk(),nn.Linear(128,48))
        self.classifier=nn.Sequential(nn.Linear(48,64),nn.SiLU(),nn.Linear(64,4))
        self.decoder=nn.Sequential(nn.Linear(48,128),nn.SiLU(),nn.Linear(128,256),nn.SiLU(),nn.Linear(256,512))
    def encode(self,x):return self.encoder(x)
    def decode(self,z):return self.decoder(z).reshape(-1,1,512)

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
        loss=ce+rec+args.contrast_weight*cl+args.cycle_weight*cycle
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%300==0 or step==args.representation_steps:
            row={'step':step,'ce':float(ce.detach()),'reconstruction':float(rec.detach()),'source_contrastive':float(cl.detach()),'latent_cycle':float(cycle.detach())}
            history.append(row);print({'phase':'representation',**row},flush=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    base.dump(out/'representation_training.json',history);return model
