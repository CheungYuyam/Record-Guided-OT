"""Coupling-controlled conditional Brownian diffusion bridges.

Learned endpoint couplings are exploratory UDA choices, not an exact SB claim.
"""
import math
import torch
from torch import nn
from models import base
from core import seed_all

ARMS=['standard_ddim','random_bridge','distance_ot_bridge','semantic_ot_bridge','semantic_ot_barycentric']
SIGMA=.5

class EndpointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(48+48+9+32,256),nn.SiLU(),
                               nn.Linear(256,256),nn.SiLU(),nn.Linear(256,48))
    def forward(self,x,source,condition,t):
        f=torch.exp(torch.arange(16,device=x.device)*(-math.log(10000)/15))
        phase=t[:,None]*1000*f[None]
        return source+self.net(torch.cat([x,source,condition,phase.sin(),phase.cos()],1))

@torch.no_grad()
def fit_temperature(logits,y):
    # Only labeled source validation data; finite fixed grid, never target scores.
    grid=torch.tensor([.5,.75,1.,1.5,2.,3.,4.],device=logits.device)
    losses=torch.stack([nn.functional.cross_entropy(logits/t,y) for t in grid])
    i=int(losses.argmin())
    return float(grid[i]),{'grid':grid.tolist(),'source_validation_NLL':losses.tolist(),'selected':float(grid[i])}

@torch.no_grad()
def sinkhorn(cost,epsilon=.1,max_steps=20000):
    n,m=cost.shape
    logk=-cost.double()/epsilon
    loga=torch.full((n,),-math.log(n),device=cost.device,dtype=torch.float64)
    logb=torch.full((m,),-math.log(m),device=cost.device,dtype=torch.float64)
    u=torch.zeros_like(loga);v=torch.zeros_like(logb)
    for step in range(max_steps):
        u=loga-torch.logsumexp(logk+v[None],1)
        v=logb-torch.logsumexp(logk+u[:,None],0)
        if (step+1)%25==0:
            logp=logk+u[:,None]+v[None]
            error=max(float((torch.exp(torch.logsumexp(logp,1)-loga)-1).abs().max()),
                      float((torch.exp(torch.logsumexp(logp,0)-logb)-1).abs().max()))
            if error<.001:break
    assert error<.001,('Sinkhorn did not meet frozen relative marginal tolerance',error)
    plan=logp.exp().float()
    return plan,{'iterations':step+1,'max_relative_marginal_error':error,'epsilon':epsilon}

@torch.no_grad()
def plans(us,ut,y,target_prob):
    distance=torch.cdist(us,ut).square()/us.shape[1]
    scale=distance.median().clamp_min(1e-6)
    distance=distance/scale
    # Entropy shrinkage limits influence of uncertain pseudo-label distributions.
    entropy=-(target_prob*target_prob.clamp_min(1e-8).log()).sum(1)/math.log(9)
    reliability=(1-entropy).clamp(0,1)
    shrunk=reliability[:,None]*target_prob+(1-reliability[:,None])/9
    semantic=-shrunk[:,y].T.clamp_min(1e-8).log()
    semscale=semantic.median().clamp_min(1e-6)
    result={};audit={'distance_scale':float(scale),'semantic_scale':float(semscale),
                    'target_mean_reliability':float(reliability.mean()),
                    'target_soft_class_mass':target_prob.mean(0).tolist()}
    for name,cost in [('distance_ot_bridge',distance),('semantic_ot_bridge',distance+semantic/semscale)]:
        plan,checks=sinkhorn(cost)
        result[name]=plan
        audit[name]={**checks,'expected_normalized_distance':float((plan*distance).sum()),
            'expected_source_label_pseudo_probability':float((plan*target_prob[:,y].T).sum()),
            'row_entropy_mean':float((-(plan*len(us))*(plan*len(us)).clamp_min(1e-12).log()).sum(1).mean())}
    return result,audit

def train_bridge(mode,us,ut,y,plan,args,out):
    seed_all(args.seed+700);net=EndpointNet().to(us.device)
    opt=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001)
    gen=torch.Generator(device=us.device).manual_seed(args.seed+701)
    history=[]
    for step in range(1,args.bridge_steps+1):
        si=base.sample(len(us),64,us.device,gen)
        if plan is None:ti=base.sample(len(ut),64,ut.device,gen)
        else:ti=torch.multinomial(plan[si],1,generator=gen).squeeze(1)
        source=us[si];target=ut[ti];condition=nn.functional.one_hot(y[si],9).float()
        t=torch.rand((64,),device=us.device,generator=gen).clamp_min(.001)
        noise=torch.randn(source.shape,device=us.device,generator=gen)
        x=(1-t[:,None])*target+t[:,None]*source+SIGMA*(t*(1-t)).sqrt()[:,None]*noise
        loss=nn.functional.mse_loss(net(x,source,condition,t),target)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if step%400==0 or step==args.bridge_steps:
            history.append({'step':step,'target_latent_MSE':float(loss.detach())})
            print({'phase':mode,'step':step,'loss':float(loss.detach())},flush=True)
    net.eval()
    for p in net.parameters():p.requires_grad_(False)
    torch.save({'state_dict':net.state_dict(),'step':args.bridge_steps,'sigma':SIGMA,'mode':mode},out/f'{mode}_generator_last.pt')
    base.dump(out/f'{mode}_generator_training.json',history)
    return net

def posterior(x,endpoint,t,s,noise,sigma=SIGMA):
    return (1-s/t)*endpoint+(s/t)*x+sigma*math.sqrt(max(s*(t-s)/t,0))*noise

@torch.no_grad()
def generate(mode,net,source,condition,ab,seed,steps=32):
    gen=torch.Generator(device=source.device).manual_seed(seed)
    paths=[source.cpu()];traces=[]
    if mode=='standard_ddim':
        times=torch.linspace(150,0,steps,device=source.device).round().long()
        noise=torch.randn(source.shape,device=source.device,generator=gen)
        state=ab[times[0]].sqrt()*source+(1-ab[times[0]]).sqrt()*noise
        for k,t in enumerate(times):
            pred=net(state,t.expand(len(state)),condition,torch.ones(len(state),device=source.device,dtype=torch.long))
            estimate=(state-(1-ab[t]).sqrt()*pred)/ab[t].sqrt()
            previous=ab[times[k+1]] if k+1<len(times) else torch.ones((),device=state.device)
            before=state;state=previous.sqrt()*estimate+(1-previous).sqrt()*pred
            assert torch.isfinite(state).all()
            paths.append(state.cpu());traces.append({'step':k+1,'time':int(t),'step_RMS':float((state-before).square().mean().sqrt())})
    else:
        state=source.clone()
        for k in range(steps):
            t=1-k/steps;s=1-(k+1)/steps
            estimate=net(state,source,condition,torch.full((len(state),),t,device=state.device))
            noise=torch.randn(source.shape,device=source.device,generator=gen)
            before=state;state=posterior(state,estimate,t,s,noise)
            assert torch.isfinite(state).all()
            paths.append(state.cpu());traces.append({'step':k+1,'time':t,'step_RMS':float((state-before).square().mean().sqrt())})
    return state,traces,torch.stack(paths)

@torch.no_grad()
def verify(device):
    cost=torch.full((9,9),5.,device=device);cost.fill_diagonal_(0)
    p,a=sinkhorn(cost)
    assert float(p.diagonal().sum())>.999
    g=torch.Generator(device=device).manual_seed(1601)
    source=torch.tensor([2.,-1.],device=device).expand(20000,-1)
    endpoint=torch.tensor([-2.,3.],device=device).expand_as(source)
    x=posterior(source,endpoint,1.,.5,torch.randn(source.shape,device=device,generator=g))
    expected=.5*(source[0]+endpoint[0])
    mean_error=float((x.mean(0)-expected).abs().max())
    variance_error=float((x.var(0)-SIGMA**2*.25).abs().max())
    assert mean_error<.01 and variance_error<.003
    final=posterior(x,endpoint,.5,0.,torch.zeros_like(x))
    assert torch.equal(final,endpoint)
    return {'success':True,'sinkhorn':a,'bridge_posterior_mean_error':mean_error,
            'bridge_posterior_variance_error':variance_error,'oracle_endpoint_exact':True}
