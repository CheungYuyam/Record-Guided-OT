import numpy as np
import torch
from torch.nn import functional as F
from models import encode,base
from bridges import fit_temperature
@torch.no_grad()
def transport_labels(plan,y):
    # This is exactly the joint law of V18's sampler: uniform source, conditional target.
    joint=plan/plan.sum(1,keepdim=True)/len(plan)
    mass=joint.sum(0)
    soft=(joint.T@F.one_hot(y,9).float())/mass[:,None]
    assert torch.isfinite(soft).all() and (soft>=0).all()
    assert torch.allclose(soft.sum(1),torch.ones_like(mass),atol=1e-6)
    return soft,mass/mass.sum()
@torch.no_grad()
def probabilities(model,x,xv,yv):
    model.eval()
    temp,cal=fit_temperature(model.classifier(encode(model,xv)),yv)
    return (model.classifier(encode(model,x))/temp).softmax(-1),cal

@torch.no_grad()
def save_adaptation_snapshot(arm,step,model,plan,y,xt,xv,yv,rid,out):
    soft,mass=transport_labels(plan,y)
    student,cal=probabilities(model,xt,xv,yv)
    path=out/f'{arm}_step{step:04d}_adaptation_predictions.npz'
    np.savez_compressed(path,record_id=rid.cpu().numpy(),transport_probability=soft.cpu().numpy(),
        student_probability=student.cpu().numpy(),
        target_sampling_mass=mass.cpu().numpy())
    return {'path':path.name,'sha256':base.sha(path),'contains_target_truth':False,
            'student_source_calibration':cal}
