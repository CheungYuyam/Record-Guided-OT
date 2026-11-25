"""Alternating representation/coupling updates; no target truth inputs."""
import copy, math, hashlib
import numpy as np
import torch
from torch.nn import functional as F
from models import Representation, encode, base
from core import seed_all, supcon
from bridges import sinkhorn, fit_temperature
from v20_geometry import geometry_cost, paired_distance, verify_geometry
from snapshots import save_adaptation_snapshot

ARMS = ['primary', 'opposite_semantic', 'no_record', 'no_record_half', 'record_mean_only', 'shuffled_records', 'fixed_coupling', 'no_ot', 'weight_025', 'weight_075', 'epsilon_005', 'epsilon_020', 'refresh_150', 'refresh_600']
ARMS += ['ce_zero', 'ce_005', 'ce_010', 'align_zero', 'align_0025', 'align_005', 'joint_weak', 'ramp_600']
PAIR_ALIGN = .1
PAIR_CE = .2
ARMS += ['record_zero', 'record_one', 'entropy_003', 'entropy_030', 'metric_010', 'metric_050', 'semantic_025', 'semantic_100']
USE_METRIC = False
SEMANTIC_WEIGHT = 0.
ARMS += ['baseline_replay','record_one_ce005']
RECORD_ARMS = set(ARMS)-{"no_record","no_record_half"}
PRIMARY_SEMANTIC = False

def semantic_enabled(arm):
    return True if arm=="opposite_semantic" else SEMANTIC_WEIGHT>0


def shuffle_record_groups(rid, seed):
    # Label-blind fixed permutation of assignments; exactly preserves group sizes.
    g=torch.Generator(device=rid.device).manual_seed(seed)
    return rid[torch.randperm(len(rid),device=rid.device,generator=g)]

def coupling_cost(cost, semantic, semscale, arm):
    return cost+(1. if arm=='opposite_semantic' else SEMANTIC_WEIGHT)*semantic/semscale if semantic_enabled(arm) else cost


@torch.no_grad()
def make_plan(model, xs, y, xt, xv, yv, teacher_probability, lm, ls, arm="v18_replay", source_rid=None, target_rid=None):
    model.eval()
    us, ut = (encode(model, xs)-lm)/ls, (encode(model, xt)-lm)/ls
    temp, calibration = fit_temperature(model.classifier(encode(model, xv)), yv)
    q = .5*(model.classifier(ut*ls+lm)/temp).softmax(-1)+.5*teacher_probability
    cost,transform,geometry=geometry_cost(us,ut,y,source_rid,target_rid,
        metric=USE_METRIC,record=arm in RECORD_ARMS,mean_only=arm=='record_mean_only')
    if arm=='no_record_half':
        cost=0.5*cost
        geometry['window_cost_coefficient']=0.5
    scale=geometry['distance_scale']
    reliability = (1+(q*q.clamp_min(1e-8).log()).sum(1)/math.log(4)).clamp(0,1)
    shrunk = reliability[:,None]*q+(1-reliability[:,None])/4
    semantic = -shrunk[:,y].T.clamp_min(1e-8).log()
    semscale = semantic.median().clamp_min(1e-6)
    plan, audit = sinkhorn(coupling_cost(cost,semantic,semscale,arm))
    audit["semantic_cost_enabled"]=semantic_enabled(arm)
    audit.update(distance_scale=float(scale), semantic_scale=float(semscale),
                 representation_calibration=calibration,
                 target_soft_class_mass=q.mean(0).tolist(),
                 target_mean_reliability=float(reliability.mean()),
                 expected_normalized_geometry_cost=float((plan*cost).sum()))
    audit.update(geometry)
    return plan, audit, transform

def ot_loss(zs, zt, labels, logits, ls, distance_scale, transform=None):
    # Samples follow P(i,j): uniform source then conditional target.
    distance = ((zs-zt)/ls).square().mean()/distance_scale if transform is None else paired_distance(zs,zt,ls,transform,distance_scale)
    semantic = F.cross_entropy(logits, labels)
    return PAIR_ALIGN*distance+PAIR_CE*semantic, distance, semantic

def refine(arm, initial, initial_plan, initial_scale, xs, y, xt, xv, yv,
           teacher_probability, lm, ls, args, out, source_rid, target_rid):
    seed_all(args.seed+1800)
    model=copy.deepcopy(initial)
    for p in model.parameters(): p.requires_grad_(True)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
    g=torch.Generator(device=xs.device).manual_seed(args.seed+1801)
    gp=torch.Generator(device=xs.device).manual_seed(args.seed+1802)
    plan=initial_plan.clone();scale=initial_scale
    geom_srid,geom_trid=source_rid,target_rid
    group_audit={'shuffled':False}
    if arm=='shuffled_records':
        geom_srid=shuffle_record_groups(source_rid,args.seed+2101)
        geom_trid=shuffle_record_groups(target_rid,args.seed+2102)
        for actual,shuffled in [(source_rid,geom_srid),(target_rid,geom_trid)]:
            assert torch.equal(actual.sort().values,shuffled.sort().values)
            assert not torch.equal(actual,shuffled)
        def digest(rid):return hashlib.sha256(rid.cpu().numpy().tobytes()).hexdigest()
        group_audit={'shuffled':True,'target_labels_used':False,'fixed_across_refreshes':True,
                     'source_seed':args.seed+2101,'target_seed':args.seed+2102,
                     'source_assignment_sha256':digest(geom_srid),'target_assignment_sha256':digest(geom_trid)}
        np.savez_compressed(out/f'{arm}_group_assignments.npz',source_record_id=source_rid.cpu().numpy(),
            target_record_id=target_rid.cpu().numpy(),source_geometry_group=geom_srid.cpu().numpy(),target_geometry_group=geom_trid.cpu().numpy())
    transform=None;initial_geometry=None
    if arm!='v18_replay':
        plan,initial_geometry,transform=make_plan(model,xs,y,xt,xv,yv,teacher_probability,lm,ls,arm,geom_srid,geom_trid)
        scale=initial_geometry['distance_scale']
        initial_geometry['L1_change_from_V16']=float((plan-initial_plan).abs().sum())
    history=[];refresh=[]
    snapshots=[save_adaptation_snapshot(arm,0,model,plan,y,xt,xv,yv,target_rid,out)]
    for step in range(1,args.refinement_steps+1):
        model.train()
        si=base.balanced(y,g);ti=base.sample(len(xt),len(si),xt.device,g)
        x=torch.cat([xs[si],xt[ti]]);z=model.encode(x);n=len(si)
        reconstruction=model.decode(z)
        ce=F.cross_entropy(model.classifier(z[:n]),y[si])
        rec=F.mse_loss(reconstruction,x)
        cycle=F.mse_loss(model.encode(reconstruction),z.detach())
        contrast=supcon(F.normalize(z[:n],dim=-1),y[si])
        loss=ce+rec+args.cycle_weight*cycle+args.contrast_weight*contrast
        d=ce.detach()*0;s=d
        if arm!='no_ot':
            oi=base.sample(len(xs),64,xs.device,gp)
            oj=torch.multinomial(plan[oi],1,generator=gp).squeeze(1)
            zs=model.encode(xs[oi]);zt=model.encode(xt[oj])
            extra,d,s=ot_loss(zs,zt,y[oi],model.classifier(zt),ls,scale,transform)
            loss=loss+min(step/args.ramp_steps,1)*extra
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%300==0 or step==args.refinement_steps:
            row={'step':step,'source_CE':float(ce.detach()),'reconstruction':float(rec.detach()),
                 'cycle':float(cycle.detach()),'contrast':float(contrast.detach()),
                 'OT_distance':float(d.detach()),'OT_target_CE':float(s.detach())}
            history.append(row);print({'arm':arm,**row},flush=True)
        if step%args.refresh_interval==0 or step==args.refinement_steps:
            if arm in ['fixed_coupling','no_ot']:
                new,audit,new_transform=plan,dict(initial_geometry),transform
                audit['frozen_step_zero_coupling']=True
            else:
                new,audit,new_transform=make_plan(model,xs,y,xt,xv,yv,teacher_probability,lm,ls,arm,geom_srid,geom_trid)
            transform=None if arm=='v18_replay' else new_transform
            audit.update(step=step,L1_change=float((new-plan).abs().sum()))
            refresh.append(audit);plan=new;scale=audit['distance_scale']
            snapshots.append(save_adaptation_snapshot(arm,step,model,plan,y,xt,xv,yv,target_rid,out))
            base.dump(out/f'{arm}_refresh.json',refresh)
            print({'arm':arm,'refresh_step':step,'L1_change':audit['L1_change']},flush=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    changes={name:float(sum((v-before[k]).square().sum() for k,v in model.state_dict().items()
                          if k.startswith(name)).sqrt()) for name in ['encoder','classifier','decoder']}
    assert all(v>0 for v in changes.values())
    assert len(refresh)>0
    assert all(row['L1_change']==0 for row in refresh) if arm in ['fixed_coupling','no_ot'] else any(row['L1_change']>0 for row in refresh)
    path=out/f'{arm}_representation_last.pt'
    torch.save({'state_dict':model.state_dict(),'step':args.refinement_steps,
                'latent_mean':lm,'latent_std':ls},path)
    reloaded=Representation().to(xs.device)
    reloaded.load_state_dict(torch.load(path,map_location=xs.device,weights_only=False)['state_dict'])
    reloaded.eval()
    with torch.no_grad():
        a=model.encode(xv);b=reloaded.encode(xv)
        error=float((a-b).abs().max())
        decode_error=float((model.decode(a)-reloaded.decode(b)).abs().max())
    assert error==0 and decode_error==0
    np.savez_compressed(out/f'{arm}_final_coupling.npz',plan=plan.cpu().numpy(),source_record_id=source_rid.cpu().numpy(),target_record_id=target_rid.cpu().numpy())
    base.dump(out/f'{arm}_refinement.json',{'history':history,'refresh':refresh,
        'parameter_L2_change':changes,'representation_reload_error':error,
        'decoder_reload_error':decode_error,'weight_sha256':base.sha(path),
        'coupling_sha256':base.sha(out/f'{arm}_final_coupling.npz'),
        'target_labels_used':False,'refinement_steps':args.refinement_steps,
        'initial_plan_preserved':False,'initial_geometry':initial_geometry,'snapshots':snapshots,
        'metric_enabled':False,'record_enabled':arm in RECORD_ARMS,
        'semantic_cost_enabled':semantic_enabled(arm),'coupling_refresh_enabled':arm not in ['fixed_coupling','no_ot'],'OT_training_loss_enabled':arm!='no_ot','group_audit':group_audit})
    return reloaded,plan

def verify(device):
    # Semantic coupling must move when geometry swaps, and OT loss must reach both encoders/head.
    c=torch.tensor([[0.,4.],[4.,0.]],device=device)
    a,_=sinkhorn(c);b,_=sinkhorn(c.flip(1))
    assert a.diagonal().sum()>.999 and b.flip(1).diagonal().sum()>.999
    zs=torch.tensor([[1.,2.],[3.,4.]],device=device,requires_grad=True)
    zt=torch.tensor([[2.,4.],[1.,2.]],device=device,requires_grad=True)
    head=torch.nn.Linear(2,2).to(device)
    loss,_,_=ot_loss(zs,zt,torch.tensor([0,1],device=device),head(zt),torch.ones(2,device=device),1.)
    loss.backward()
    assert zs.grad.abs().sum()>0 and zt.grad.abs().sum()>0 and head.weight.grad.abs().sum()>0
    return {'success':True,'coupling_geometry_sensitive':True,
            'OT_gradients_reach_source_target_and_head':True,**verify_geometry(device),**verify_ablation_controls(device)}


def verify_ablation_controls(device):
    import importlib.util
    from pathlib import Path
    # Verify main geometry against its frozen, unchanged V20 implementation.
    original=Path(__file__).resolve().parent/'frozen_v20_geometry.py'
    spec=importlib.util.spec_from_file_location('frozen_v20_geometry_reference',original)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    g=torch.Generator(device=device).manual_seed(21000)
    a=torch.randn(24,4,device=device,generator=g);b=torch.randn(16,4,device=device,generator=g)
    sr=torch.arange(24,device=device)//4;tr=torch.arange(16,device=device)//4;y=sr%3
    actual,_,_=geometry_cost(a,b,y,sr,tr,record=True)
    expected,_,_=reference.geometry_cost(a,b,y,sr,tr,record=True)
    assert torch.equal(actual,expected)
    from v20_geometry import record_moments
    sm,_,si,_=record_moments(a,sr);tm,_,ti,_=record_moments(b,tr)
    point=torch.cdist(a,b).square()/4;record=torch.cdist(sm,tm).square()/4
    expected_mean=.5*point/point.median()+.5*(record/record.median())[si[:,None],ti[None,:]]
    mean,_,_=geometry_cost(a,b,y,sr,tr,record=True,mean_only=True)
    assert torch.allclose(mean,expected_mean,atol=1e-6)
    shuffled=shuffle_record_groups(sr,2101)
    assert torch.equal(shuffled,shuffle_record_groups(sr,2101))
    assert torch.equal(sr.sort().values,shuffled.sort().values) and not torch.equal(sr,shuffled)
    semantic=torch.randn_like(actual)
    assert torch.equal(coupling_cost(actual,semantic,2.,'no_semantic_cost'),actual)
    assert torch.equal(coupling_cost(actual,semantic*7,2.,'no_semantic_cost'),actual)
    return {'V20_main_geometry_bitwise_equal':True,'mean_only_formula_checked':True,
            'shuffle_deterministic_size_preserving':True,'semantic_cost_disabled_checked':True}
