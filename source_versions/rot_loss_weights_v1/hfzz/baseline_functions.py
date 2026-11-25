"""Unified-backbone UDA baselines; not original image-benchmark reproductions."""
import itertools
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from core import Classifier, reverse, seed_all
from models import base

ARMS = ['distance_ot_barycentric', 'pseudo_label', 'cdan', 'cdan_e', 'deepjdot']

def joint_map(h, logits):
    # Matches author CDAN implementation: detach probabilities in outer product.
    return (logits.softmax(-1).detach()[:, :, None] * h[:, None, :]).flatten(1)

def conditional_loss(discriminator, h, logits, ramp, entropy_weighted):
    domain_logits = discriminator(reverse(joint_map(h, logits), ramp)).flatten()
    n = len(h)//2
    domain_labels = torch.cat([torch.ones(n, device=h.device), torch.zeros(n, device=h.device)])
    losses = F.binary_cross_entropy_with_logits(domain_logits, domain_labels, reduction='none')
    if not entropy_weighted:
        return losses.mean()
    p = logits.softmax(-1)
    entropy = -(p * (p+1e-5).log()).sum(-1)
    # Author CDAN+E: reverse entropy gradient, normalize weights by each domain;
    # normalization denominators detached.
    weight = 1+torch.exp(-reverse(entropy, ramp))
    weight = torch.cat([weight[:n]/weight[:n].sum().detach(),
                        weight[n:]/weight[n:].sum().detach()])
    return (weight*losses).sum()/weight.sum().detach()

def joint_cost(hs, ht, labels, target_logits):
    # Eq. (8)/(9) DeepJDOT; alpha=1/128, lambda_t=1, fixed for this HFZZ run.
    distance = torch.cdist(hs, ht).square()/hs.shape[1]
    classification = -target_logits.log_softmax(-1)[:, labels].T
    return distance+classification

def exact_assignment(cost):
    # Equal-size, uniform-mass OT has a permutation optimum (Birkhoff).
    # Hungarian assignment is therefore an exact unregularized OT solver here.
    row, col = linear_sum_assignment(cost.detach().double().cpu().numpy())
    return torch.as_tensor(row, device=cost.device), torch.as_tensor(col, device=cost.device)

def train(arm, xs, y, xt, pseudo_prob, args, out, mean, std):
    seed_all(args.seed)
    model = Classifier().to(xs.device)
    domain = nn.Sequential(nn.Linear(128*4, 64), nn.SiLU(), nn.Linear(64, 1)).to(xs.device)
    params = list(model.parameters())+(list(domain.parameters()) if arm in ['cdan','cdan_e'] else [])
    opt = torch.optim.AdamW(params, lr=.001, weight_decay=.0001)
    gi = torch.Generator(device=xs.device).manual_seed(args.seed+1)
    gp = torch.Generator(device=xs.device).manual_seed(args.seed+2)
    confidence, pseudo = pseudo_prob.max(-1)
    accepted = torch.where(confidence >= .9)[0]
    history = []; start = time.time()
    for step in range(1, args.classifier_steps+1):
        model.train()
        si = base.sample(len(xs), 64, xs.device, gi)
        details = {}
        if arm == 'pseudo_label':
            if len(accepted):
                ti = accepted[base.sample(len(accepted),32,xs.device,gp)]
                xx = torch.cat([xs[si[:32]],xt[ti]])
                yy = torch.cat([y[si[:32]],pseudo[ti]])
            else:
                xx, yy = xs[si], y[si]
            loss = F.cross_entropy(model(xx),yy)
            details['accepted_target_fraction'] = float(len(accepted)/len(xt))
        else:
            ti = base.sample(len(xt),64,xs.device,gp)
            h = model.encoder(torch.cat([xs[si],xt[ti]]))
            logits = model.head(h)
            ce = F.cross_entropy(logits[:64],y[si])
            if arm in ['cdan','cdan_e']:
                adaptation = conditional_loss(domain,h,logits,min(step/500,1.),arm=='cdan_e')
                loss = ce+.2*adaptation
            elif arm == 'deepjdot':
                cost = joint_cost(h[:64],h[64:],y[si],logits[64:])
                ri,ci = exact_assignment(cost)
                adaptation = cost[ri,ci].mean()
                loss = ce+adaptation
            else:
                raise ValueError(arm)
            details={'source_CE':float(ce.detach()),'adaptation':float(adaptation.detach())}
        assert torch.isfinite(loss), (arm,step)
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(params,5);opt.step()
        if step%400==0 or step==args.classifier_steps:
            row={'step':step,'loss':float(loss.detach()),'seconds':time.time()-start,**details}
            history.append(row);print({'arm':arm,**row},flush=True)
    model.eval()
    torch.save({'state_dict':model.state_dict(),'step':args.classifier_steps,
                'mean':mean.cpu(),'std':std.cpu()},out/f'{arm}_last.pt')
    if arm in ['cdan','cdan_e']:
        torch.save({'state_dict':domain.state_dict(),'step':args.classifier_steps},
                   out/f'{arm}_domain_last.pt')
    base.dump(out/f'{arm}_training.json',history)
    base.dump(out/f'{arm}_budget.json',{'classifier_updates':args.classifier_steps,
              'trainable_parameters':sum(p.numel() for p in params),'seconds':time.time()-start,
              'source_batch':32 if arm=='pseudo_label' and len(accepted) else 64,
              'target_batch':32 if arm=='pseudo_label' and len(accepted) else (0 if arm=='pseudo_label' else 64),
              'uses_frozen_teacher_ensemble':arm=='pseudo_label',
              'label':'unified CNN/spectral-input implementation; not original-paper architecture or tuned reproduction'})
    return model

def verify(device):
    # Nontrivial OT matching tested against exhaustive optimum on a small case.
    cost=torch.tensor([[4.,1.,7.,3.],[2.,9.,0.,6.],[8.,5.,3.,2.],[1.,4.,6.,8.]],device=device)
    ri,ci=exact_assignment(cost)
    optimum=min(sum(float(cost[i,p[i]]) for i in range(4)) for p in itertools.permutations(range(4)))
    assert abs(float(cost[ri,ci].sum())-optimum)<1e-7
    x=torch.tensor([1.,2.],device=device,requires_grad=True)
    reverse(x,.3).sum().backward()
    assert torch.allclose(x.grad,torch.full_like(x,-.3))
    h=torch.randn(8,128,device=device,requires_grad=True)
    logits=torch.randn(8,4,device=device,requires_grad=True)
    joint_map(h,logits).sum().backward()
    assert h.grad is not None and logits.grad is None
    d=nn.Linear(512,1).to(device)
    for weighted in [False,True]:
        hh=h.detach().clone().requires_grad_()
        ll=logits.detach().clone().requires_grad_()
        v=conditional_loss(d,hh,ll,.5,weighted)
        v.backward()
        assert torch.isfinite(v) and torch.isfinite(hh.grad).all()
        if weighted:assert ll.grad is not None and torch.isfinite(ll.grad).all()
    hs=torch.randn(4,128,device=device,requires_grad=True)
    ht=torch.randn(4,128,device=device,requires_grad=True)
    lt=torch.randn(4,4,device=device,requires_grad=True)
    jc=joint_cost(hs,ht,torch.arange(4,device=device),lt)
    ri,ci=exact_assignment(jc);jc[ri,ci].mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in [hs,ht,lt])
    return {'success':True,'exact_uniform_OT_matches_bruteforce':True,
            'gradient_reversal_sign':True,'CDAN_outer_product_probability_detached':True,
            'CDAN_entropy_gradient_finite':True,'DeepJDOT_feature_and_target_label_gradients_finite':True}
