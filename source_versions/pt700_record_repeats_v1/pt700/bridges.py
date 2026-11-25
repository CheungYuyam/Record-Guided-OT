import math
import torch
from torch import nn
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
    entropy=-(target_prob*target_prob.clamp_min(1e-8).log()).sum(1)/math.log(4)
    reliability=(1-entropy).clamp(0,1)
    shrunk=reliability[:,None]*target_prob+(1-reliability[:,None])/4
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
