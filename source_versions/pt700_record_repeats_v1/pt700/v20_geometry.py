"""V20-specific module name avoids collision with the frozen V14 geometry module."""
import torch

SHRINKAGE=.1
RECORD_WEIGHT=.5

@torch.no_grad()
def source_metric(u,y):
    centered=torch.empty_like(u,dtype=torch.float64)
    for c in y.unique():
        ix=y==c;v=u[ix].double();centered[ix]=v-v.mean(0)
    cov=centered.T@centered/len(u);d=u.shape[1]
    tau=cov.trace().div(d).clamp_min(1e-8)
    cov=(1-SHRINKAGE)*cov+SHRINKAGE*tau*torch.eye(d,device=u.device,dtype=torch.float64)
    ev,vec=torch.linalg.eigh(cov)
    inv=ev.clamp_min(tau*SHRINKAGE).reciprocal();inv=inv*d/inv.sum()
    # Symmetric square root: squared distance of u @ transform is Mahalanobis.
    transform=(vec*inv.sqrt()[None,:])@vec.T
    return transform.to(u.dtype),{'shrinkage':SHRINKAGE,'fit_split':'source_train',
        'covariance_min_eigenvalue':float(ev.min()),'metric_condition_number':float(inv.max()/inv.min()),
        'metric_trace':float(inv.sum())}

def groups(rid):
    ids,ix,counts=torch.unique(rid,sorted=True,return_inverse=True,return_counts=True)
    return ix,counts

@torch.no_grad()
def record_moments(u,rid):
    ix,count=groups(rid)
    mean=torch.zeros(len(count),u.shape[1],device=u.device,dtype=u.dtype)
    mean.index_add_(0,ix,u);mean/=count[:,None]
    var=torch.zeros_like(mean);var.index_add_(0,ix,(u-mean[ix]).square());var/=count[:,None]
    return mean,var.clamp_min(0).sqrt(),ix,count

@torch.no_grad()
def geometry_cost(us,ut,y,srid,trid,metric=False,record=False,mean_only=False):
    d=us.shape[1];transform=torch.eye(d,device=us.device,dtype=us.dtype)
    audit={'metric_enabled':metric,'record_enabled':record}
    if metric:transform,ma=source_metric(us,y);audit.update(ma)
    a,b=us@transform,ut@transform
    distance=torch.cdist(a,b).square()/d
    scale=distance.median().clamp_min(1e-6)
    cost=distance/scale
    audit['distance_scale']=float(scale)
    if record:
        sm,ss,si,sc=record_moments(a,srid);tm,ts,ti,tc=record_moments(b,trid)
        # Exact W2^2 for the diagonal Gaussian approximations in this coordinate system.
        rd=(torch.cdist(sm,tm).square()+torch.cdist(ss,ts).square())/d
        if mean_only: rd=torch.cdist(sm,tm).square()/d
        audit["record_mean_only"]=mean_only
        rs=rd.median().clamp_min(1e-6)
        cost=(1-RECORD_WEIGHT)*cost+RECORD_WEIGHT*(rd/rs)[si[:,None],ti[None,:]]
        audit.update(record_weight=RECORD_WEIGHT,record_distance_scale=float(rs),
                     source_records=len(sc),target_records=len(tc),
                     source_record_windows_min=int(sc.min()),target_record_windows_min=int(tc.min()))
    return cost,transform,audit

def paired_distance(zs,zt,ls,transform,scale):
    return (((zs-zt)/ls)@transform).square().mean()/scale

def verify_geometry(device):
    g=torch.Generator(device=device).manual_seed(20261021)
    x=torch.randn(24,4,device=device,generator=g);y=torch.arange(24,device=device)//8
    rid=torch.arange(24,device=device)//4
    transform,audit=source_metric(x,y);m=transform@transform.T
    assert torch.linalg.eigvalsh(m).min()>0 and abs(float(m.trace())-4)<1e-5
    a=x[:7];b=x[7:14];delta=a-b
    err=float(((delta@transform).square().sum(1)-(delta*(delta@m)).sum(1)).abs().max())
    assert err<1e-5
    # Record and point costs are invariant to window order / arbitrary record ID renaming.
    cost,_,_=geometry_cost(x,x.flip(0),y,rid,rid,True,True)
    perm=torch.randperm(24,device=device,generator=g)
    new,_,_=geometry_cost(x[perm],x.flip(0),y[perm],100-rid[perm],500+rid*13,True,True)
    perm_error=float((new-cost[perm]).abs().max());assert perm_error<1e-4
    mean,std,ix,c=record_moments(x,rid)
    assert torch.allclose(mean,torch.stack([x[rid==r].mean(0) for r in rid.unique()]),atol=1e-6)
    # One-dimensional Gaussian closed form: (mu1-mu2)^2+(sigma1-sigma2)^2.
    xx=torch.tensor([[-1.],[1.],[1.],[5.]],device=device)
    mm,sd,_,_=record_moments(xx,torch.tensor([0,0,1,1],device=device))
    w2=(mm[0]-mm[1]).square()+(sd[0]-sd[1]).square();assert float(w2)==10
    zs=x[:7].clone().requires_grad_();zt=x[7:14].clone().requires_grad_()
    paired_distance(zs,zt,torch.ones(4,device=device),transform,1.).backward()
    assert zs.grad.abs().sum()>0 and zt.grad.abs().sum()>0
    return {'SPD_and_trace_checked':True,'Mahalanobis_formula_max_error':err,
        'record_permutation_and_ID_renaming_error':perm_error,'diagonal_Gaussian_W2_checked':True,
        'paired_metric_gradients_checked':True}
