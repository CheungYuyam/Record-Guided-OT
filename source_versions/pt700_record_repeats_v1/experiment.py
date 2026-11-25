from pathlib import Path
import sys,json,argparse,itertools,hashlib,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'pt700'))
import core,models,run as base,refinement as ref,v20_geometry as geom,bridges
from baseline_functions import conditional_loss,joint_cost,exact_assignment
from classic_reference import spectra
PAIRS=['S_to_M','S_to_L','M_to_S','M_to_L','L_to_S','L_to_M']
def read(p):return json.loads(p.read_text())
def save(p,x):base.dump(p,x)
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def prepare():
 out=ROOT/'data/pt700_record_repeats_v1';out.mkdir(exist_ok=False)
 protocol=read(HERE/'protocol.json');pool={}
 for domain in 'SML':
  sp=next(p for p in PAIRS if p[0]==domain);tp=next(p for p in PAIRS if p[-1]==domain)
  for role,pair,names in [('source',sp,['source_train','source_val']),('target',tp,['target_unlabeled','target_test'])]:
   parts=[load(ROOT/f'data/pt700_v1/{pair}/{n}.npz') for n in names]
   d={k:np.concatenate([p[k] for p in parts]) for k in ['x','record_id']}
   d['y']=(d['record_id']//100)%10
   for p in parts:
    if 'y' in p:assert np.array_equal(p['y'],(p['record_id']//100)%10)
   assert len(np.unique(d['record_id']))==140 and all(np.sum(d['record_id']==r)==16 for r in np.unique(d['record_id']))
   assert all(1<=int(r)%100<=45 for r in d['record_id'])
   pool[domain,role]=d
 manifests={}
 for repeat,splitseed in enumerate(protocol['split_seeds']):
  selections={}
  for di,domain in enumerate('SML'):
   for ri,role in enumerate(['source','target']):
    d=pool[domain,role];g=np.random.default_rng(np.random.SeedSequence([splitseed,di,ri]));first=[];second=[]
    for c in range(4):
     ids=np.unique(d['record_id'][d['y']==c]);assert len(ids)==35
     ids=g.permutation(ids);n=25 if role=='source' else 20;first.extend(ids[:n].tolist());second.extend(ids[n:].tolist())
    for name,ids in zip(['source_train','source_val'] if role=='source' else ['target_unlabeled','target_test'],[first,second]):
     mask=np.isin(d['record_id'],ids);selections[domain,name]={k:v[mask] for k,v in d.items() if k!='y' or name!='target_unlabeled'}
  for pair in PAIRS:
   dest=out/f'repeat{repeat}'/pair;dest.mkdir(parents=True)
   mf={'source_domain':pair[0],'target_domain':pair[-1],'repeat':repeat,'split_seed':splitseed,'train_seed':protocol['train_seeds'][repeat],'splits':{}}
   sets=[]
   for name in ['source_train','source_val','target_unlabeled','target_test']:
    d=selections[pair[0] if name.startswith('source') else pair[-1],name];p=dest/(name+'.npz');np.savez_compressed(p,**d)
    ids=np.unique(d['record_id']);sets.append(set(ids.tolist()));mf['splits'][name]={'sha256':base.sha(p),'record_ids':ids.tolist(),'windows':len(d['x']),'class_records':[int(np.sum((ids//100)%10==c)) for c in range(4)]}
   assert all(a.isdisjoint(b) for a,b in itertools.combinations(sets,2))
   save(dest/'manifest.json',mf);manifests[f'repeat{repeat}/{pair}']=mf
 save(out/'all_manifests.json',manifests);save(out/'FROZEN.json',{'protocol_sha256':base.sha(HERE/'protocol.json'),'manifests_sha256':base.sha(out/'all_manifests.json'),'selection_before_model_training':True})
def train_neural(arm,xs,y,xt,seed,steps,out):
 core.seed_all(seed+100);m=models.Representation().cuda();domain=nn.Sequential(nn.Linear(48 if arm=='dann' else 192,64),nn.SiLU(),nn.Linear(64,1)).cuda()
 params=list(m.encoder.parameters())+list(m.classifier.parameters())
 if arm in ['matched_cdan_e','dann']:params+=list(domain.parameters())
 opt=torch.optim.AdamW(params,lr=.001,weight_decay=.0001);g=torch.Generator(device='cuda').manual_seed(seed+101)
 for step in range(1,steps+1):
  if step==3001:opt=torch.optim.AdamW(params,lr=.0002,weight_decay=.0001)
  si=base.balanced(y,g);ti=base.sample(len(xt),len(si),'cuda',g);n=len(si);m.train()
  if arm=='matched_source':loss=F.cross_entropy(m.classifier(m.encode(xs[si])),y[si])
  else:
   z=m.encode(torch.cat([xs[si],xt[ti]]));logits=m.classifier(z);ce=F.cross_entropy(logits[:n],y[si])
   if arm=='matched_cdan_e':loss=ce+.2*conditional_loss(domain,z,logits,min(step/500,1.),True)
   elif arm=='dann':loss=ce+.2*F.binary_cross_entropy_with_logits(domain(core.reverse(z,min(step/500,1.))).flatten(),torch.cat([torch.ones(n,device='cuda'),torch.zeros(n,device='cuda')]))
   else:
    cost=joint_cost(z[:n],z[n:],y[si],logits[n:]);r,c=exact_assignment(cost);loss=ce+cost[r,c].mean()
  assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(params,5);opt.step()
  if step%1500==0:print(arm,step,float(loss),flush=True)
 m.eval();torch.save({'state_dict':m.state_dict(),'step':steps},out/f'{arm}.pt');return m
def metric(y,p,rid):
 result={}
 for unit,yy,pp in [('window',y,p),('record',np.array([y[rid==r][0] for r in np.unique(rid)]),np.stack([p[rid==r].mean(0) for r in np.unique(rid)]))]:
  pred=pp.argmax(1);cm=np.zeros((4,4),int);np.add.at(cm,(yy,pred),1);result[unit]={'accuracy_percent':float(100*np.mean(pred==yy)),'macro_f1_percent':float(100*np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1))),'confusion':cm.tolist()}
 return result
def main(a):
 p=read(HERE/'protocol.json');repeat=a.index//6;pair=PAIRS[a.index%6];seed=p['train_seeds'][repeat]
 data=ROOT/f'data/pt700_record_repeats_v1/repeat{repeat}/{pair}';mf=read(data/'manifest.json')
 out=ROOT/f'results/pt700_record_repeats_v1_{a.index:03d}{"_check" if a.check else ""}'
 if (out/'DONE.json').exists():return
 out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def checked(n):
  f=data/(n+'.npz');assert base.sha(f)==mf['splits'][n]['sha256'];return base.load(f,'cuda')
 source,val,adapt=[checked(n) for n in ['source_train','source_val','target_unlabeled']];assert 'y' not in adapt
 save(out/'run.json',{'manifest':mf,'protocol':p,'code_sha256':base.sha(Path(__file__)),'source_hashes':{f.name:base.sha(f) for f in (HERE/'pt700').glob('*.py')},'target_labels_used_for_training':False})
 fs=core.features(source['x']);mean=fs.mean((0,2),keepdim=True);std=fs.std((0,2),keepdim=True).clamp_min(1e-6);xs=(fs-mean)/std;xt=(core.features(adapt['x'])-mean)/std;y=source['y'];nets={}
 args=argparse.Namespace(seed=seed,classifier_steps=3 if a.check else 2400,representation_steps=3 if a.check else 3000,refinement_steps=3 if a.check else 1500,refresh_interval=1 if a.check else 300,ramp_steps=300,cycle_weight=1.,contrast_weight=.1)
 teacher=base.train_classifier('source_only',xs,y,xt,None,args,out,mean,std)
 initial=models.train_representation(xs,y,xt,args,out)
 with torch.no_grad():
  zz=torch.cat([models.encode(initial,xs),models.encode(initial,xt)]);lm,ls=zz.mean(0),zz.std(0).clamp_min(.05)
  xv=(core.features(val['x'])-mean)/std;temp,_=bridges.fit_temperature(teacher(xv),val['y']);tp=(teacher(xt)/temp).softmax(-1)
 geom.RECORD_WEIGHT=1.;geom.SHRINKAGE=.1;ref.PAIR_ALIGN=.1;ref.PAIR_CE=.05;ref.USE_METRIC=False;ref.SEMANTIC_WEIGHT=0.
 solver=ref.sinkhorn;ref.sinkhorn=lambda c:solver(c,epsilon=.1,max_steps=30000)
 nets['primary'],_=ref.refine('record_one_ce005',initial,torch.full((len(xs),len(xt)),1/(len(xs)*len(xt)),device='cuda'),1.,xs,y,xt,xv,val['y'],tp,lm,ls,args,out,source['record_id'],adapt['record_id'])
 for arm in ['matched_source','matched_cdan_e','matched_deepjdot','dann']:nets[arm]=train_neural(arm,xs,y,xt,seed,3 if a.check else 4500,out)
 from scipy.linalg import eigh
 from sklearn.svm import LinearSVC
 from threadpoolctl import threadpool_limits
 threadpool_limits(1)
 fs,ft=spectra(source['x'].cpu().numpy()),spectra(adapt['x'].cpu().numpy());cm=fs.mean((0,2),keepdims=True);cs=np.maximum(fs.std((0,2),ddof=1,keepdims=True),1e-6);sx=((fs-cm)/cs).reshape(len(fs),-1);tx=((ft-cm)/cs).reshape(len(ft),-1)
 sc=np.cov(sx,rowvar=False)+np.eye(sx.shape[1]);tc=np.cov(tx,rowvar=False)+np.eye(tx.shape[1]);e,v=eigh(sc);wh=(v*e**-.5)@v.T;e,v=eigh(tc);transform=wh@((v*e**.5)@v.T);assert np.allclose(transform.T@sc@transform,tc,rtol=1e-7,atol=1e-7)
 fits={}
 for arm,x in [('svm',sx),('coral_svm',sx@transform)]:
  m=LinearSVC(C=1,dual=False,tol=1e-5,max_iter=10000,random_state=seed).fit(x,y.cpu().numpy());assert m.n_iter_<10000;fits[arm]=m
  np.savez_compressed(out/f'{arm}_model.npz',coef=m.coef_,intercept=m.intercept_,mean=cm,std=cs,transform=transform)
 save(out/'TRAINING_COMPLETE.json',{'target_test_read':False,'models':list(nets)+list(fits)})
 if not a.check:
  test=checked('target_test');gy=test['y'].cpu().numpy();rid=test['record_id'].cpu().numpy();g=torch.Generator(device='cuda').manual_seed(12345+'SML'.index(pair[-1]));report={arm:{} for arm in list(nets)+list(fits)}
  for condition,snr in core.LEVELS:
   wave=test['x'] if snr is None else core.noisy(test['x'],snr,g);x=(core.features(wave)-mean)/std
   for arm,m in nets.items():
    m.eval()
    with torch.no_grad():prob=torch.cat([m.classifier(m.encode(b)).softmax(-1) for b in x.split(256)]).cpu().numpy()
    assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5);report[arm][condition]=metric(gy,prob,rid);np.savez_compressed(out/f'{arm}_{condition}.npz',y=gy,record_id=rid,score=prob)
   fx=spectra(wave.cpu().numpy());cx=((fx-cm)/cs).reshape(len(fx),-1)
   for arm,m in fits.items():
    score=m.decision_function(cx);report[arm][condition]=metric(gy,score,rid);np.savez_compressed(out/f'{arm}_{condition}.npz',y=gy,record_id=rid,score=score)
  save(out/'summary.json',report)
 save(out/'DONE.json',{'success':True,'check':a.check});print('DONE',a.index,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--index',type=int);ap.add_argument('--check',action='store_true');a=ap.parse_args()
 if a.prepare:prepare()
 else:main(a)
