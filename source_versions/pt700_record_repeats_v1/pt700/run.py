from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from core import LEVELS,Classifier,head,reverse,seed_all,features,noisy
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p,x):
    temp=Path(str(p)+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(p)

def load(p,device):
    with np.load(p) as z:return {k:torch.as_tensor(z[k],device=device,dtype=torch.float32 if k=='x' else torch.long) for k in z.files}

def short(d):
    if 'y' in d:idx=torch.cat([torch.where(d['y']==c)[0][:4] for c in range(4)])
    else:idx=torch.arange(min(36,len(d['x'])),device=d['x'].device)
    return {k:v[idx] for k,v in d.items()}

def sample(n,b,device,g):return torch.randint(n,(b,),device=device,generator=g)

def balanced(y,g):
    result=[]
    for c in range(4):
        ix=torch.where(y==c)[0];result.append(ix[sample(len(ix),4,y.device,g)])
    return torch.cat(result)

def train_classifier(arm,x,y,xt,pool,args,out,mean,std):
    seed_all(args.seed);model=Classifier().to(x.device);domain=head(128,2).to(x.device)
    params=list(model.parameters())+(list(domain.parameters()) if arm=='dann' else [])
    opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=1e-4)
    gi=torch.Generator(device=x.device).manual_seed(args.seed+1)
    gp=torch.Generator(device=x.device).manual_seed(args.seed+2);history=[];start=time.time()
    for step in range(1,args.classifier_steps+1):
        idx=sample(len(x),64,x.device,gi);xx=x[idx].clone();yy=y[idx].clone()
        if pool is not None:
            pi=sample(len(pool),32,x.device,gp);xx[32:]=pool[pi];yy[32:]=y[pi]
        if arm=='dann':
            target=xt[sample(len(xt),64,x.device,gp)]
            h=model.encoder(torch.cat([xx,target]));logits=model.head(h[:64])
            labels=torch.cat([torch.zeros(64),torch.ones(64)]).long().to(x.device)
            dl=F.cross_entropy(domain(reverse(h,min(step/500,1.))),labels)
            loss=F.cross_entropy(logits,yy)+.2*dl
        else:loss=F.cross_entropy(model(xx),yy)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite downstream classifier')
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(params,5);opt.step()
        if step%400==0 or step==args.classifier_steps:
            row={'step':step,'loss':float(loss.detach()),'seconds':time.time()-start};history.append(row)
            print(json.dumps({'classifier':arm,**row}),flush=True)
    model.eval();torch.save({'state_dict':model.state_dict(),'step':args.classifier_steps,'mean':mean.cpu(),'std':std.cpu()},out/f'{arm}_last.pt')
    dump(out/f'{arm}_training.json',history);return model

def metric(y,p):
    pred=p.argmax(1);cm=np.zeros((4,4),int);np.add.at(cm,(y,pred),1)
    return {'accuracy':float((pred==y).mean()),'macro_f1':float(np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1))),
            'per_class_recall':(cm.diagonal()/np.maximum(cm.sum(1),1)).tolist(),'confusion':cm.tolist()}

def arrays(data,mean,std,seed):
    g=torch.Generator(device=data['x'].device).manual_seed(seed)
    return {k:(features(data['x'] if snr is None else noisy(data['x'],snr,g))-mean)/std for k,snr in LEVELS}

@torch.no_grad()
def evaluate(model,inputs,data,path):
    y=data['y'].cpu().numpy();rid=data['record_id'].cpu().numpy();result={};saved={'y':y,'record_id':rid}
    ids,ix=np.unique(rid,return_inverse=True);ry=np.array([y[rid==i][0] for i in ids])
    for name,x in inputs.items():
        p=torch.cat([model(a).softmax(-1) for a in x.split(256)]).cpu().numpy();saved[name]=p
        rp=np.zeros((len(ids),4));np.add.at(rp,ix,p);rp/=np.bincount(ix)[:,None]
        result[name]={'window':metric(y,p),'record':metric(ry,rp)}
    result['mean_accuracy']=float(np.mean([result[n]['window']['accuracy'] for n,_ in LEVELS]))
    np.savez_compressed(path,**saved);return result
