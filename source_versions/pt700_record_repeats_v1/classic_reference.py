"""PT700 matched DANN and classical SVM/CORAL; final checkpoints only."""
from pathlib import Path
import argparse, sys, json, hashlib, time, warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PAIRS = ['S_to_M','S_to_L','M_to_S','M_to_L','L_to_S','L_to_M']
SEEDS = [611,622,633]
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, x): p.write_text(json.dumps(x, indent=2), encoding='utf8')
def load(p):
    with np.load(p) as z: return {k:z[k].copy() for k in z.files}
def spectra(w):
    w=w.astype(np.float64); w-=w.mean(-1,keepdims=True)
    w/=np.maximum(np.sqrt(np.mean(w*w,axis=(1,2),keepdims=True)),1e-8)
    return np.log1p(np.abs(np.fft.rfft(w,axis=-1)[...,1:])/np.sqrt(w.shape[-1]))
def evaluate(out, arm, target, prediction, condition='clean'):
    y=target['y']; cm=np.zeros((4,4),int); np.add.at(cm,(y,prediction),1)
    result={'accuracy':float(np.mean(y==prediction)), 'macro_f1':float(np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1))), 'confusion':cm.tolist()}
    np.savez_compressed(out/f'{arm}_{condition}_target_predictions.npz',y=y,prediction=prediction,record_id=target['record_id'])
    dump(out/f'{arm}_{condition}_metrics.json',result)
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--index',type=int,required=True); ap.add_argument('--check',action='store_true'); a=ap.parse_args()
    assert 0<=a.index<24
    classic=a.index>=18; pair=PAIRS[a.index-18 if classic else a.index//3]; seed=611 if classic else SEEDS[a.index%3]
    out=ROOT/'results'/f'classic_pt700_v1_{pair}_{"classic" if classic else "seed"+str(seed)}{"_check" if a.check else ""}'
    if (out/'DONE.json').exists(): print('Already complete',out,flush=True); return
    out.mkdir(exist_ok=False)
    data=ROOT/'data/pt700_v1'/pair; mf=read(data/'manifest.json')
    assert mf==read(HERE/'manifests'/f'{pair}.json')
    sets=[set(v['record_ids']) for v in mf['splits'].values()]
    assert all(not (x&y) for i,x in enumerate(sets) for y in sets[i+1:])
    def checked(name):
        p=data/(name+'.npz'); assert sha(p)==mf['splits'][name]['sha256']
        d=load(p); assert set(d['record_id'])==set(mf['splits'][name]['record_ids']); return d
    source=checked('source_train'); target=checked('target_unlabeled'); assert 'y' not in target
    dump(out/'run.json',{'pair':pair,'seed':seed,'manifest':mf,'check':a.check,'protocol':read(HERE/'protocol.json'),'code_sha256':sha(Path(__file__))})
    import torch, scipy, sklearn
    dump(out/'environment.json',{'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__})
    start=time.perf_counter()
    if classic:
        from scipy.linalg import eigh
        from sklearn.svm import LinearSVC
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
        fs,ft=spectra(source['x']),spectra(target['x']); mean=fs.mean((0,2),keepdims=True); std=np.maximum(fs.std((0,2),ddof=1,keepdims=True),1e-6)
        xs=((fs-mean)/std).reshape(len(fs),-1); xt=((ft-mean)/std).reshape(len(ft),-1)
        cs=np.cov(xs,rowvar=False)+np.eye(xs.shape[1]); ct=np.cov(xt,rowvar=False)+np.eye(xt.shape[1])
        ev,v=eigh(cs); whiten=(v*ev**-.5)@v.T
        ev,v=eigh(ct); transform=whiten@((v*ev**.5)@v.T)
        assert np.allclose(transform.T@cs@transform,ct,atol=1e-7,rtol=1e-7)
        fits={}
        for arm,x in [('svm',xs),('coral_svm',xs@transform)]:
            clf=LinearSVC(C=1,dual=False,tol=1e-5,max_iter=10000,random_state=611)
            with warnings.catch_warnings():
                warnings.simplefilter('error'); clf.fit(x,source['y'])
            assert clf.n_iter_<10000; fits[arm]=clf
            np.savez_compressed(out/f'{arm}_model.npz',coef=clf.coef_,intercept=clf.intercept_,classes=clf.classes_,mean=mean,std=std,transform=transform if arm=='coral_svm' else np.eye(xs.shape[1]))
            if arm=='coral_svm':
                np.savez_compressed(out/f'{arm}_features.npz',source=x.astype('float32'),target_adapt=xt.astype('float32'),source_y=source['y'],source_record_id=source['record_id'],target_record_id=target['record_id'])
        dump(out/'TRAINING_COMPLETE.json',{'target_test_read':False,'seconds':time.perf_counter()-start})
        if not a.check:
            test=checked('target_test')
            import torch
            sys.path.insert(0,str(HERE/'pt700'))
            import core
            raw=torch.as_tensor(test['x'],device='cuda')
            g=torch.Generator(device='cuda').manual_seed(12345+'SML'.index(pair[-1]))
            results={arm:{} for arm in fits}
            for condition,snr in core.LEVELS:
                wave=raw if snr is None else core.noisy(raw,snr,g)
                fx=spectra(wave.cpu().numpy()); tx=((fx-mean)/std).reshape(len(fx),-1)
                for arm,clf in fits.items():
                    results[arm][condition]=evaluate(out,arm,test,clf.predict(tx),condition)
            dump(out/'summary.json',results)
    else:
        import torch
        from torch import nn
        from torch.nn import functional as F
        sys.path.insert(0,str(HERE/'pt700'))
        import core, models, run as base
        torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        device='cuda'; assert torch.cuda.is_available()
        sx=torch.as_tensor(source['x'],device=device); tx=torch.as_tensor(target['x'],device=device)
        fs=core.features(sx); mean=fs.mean((0,2),keepdim=True); std=fs.std((0,2),keepdim=True).clamp_min(1e-6)
        xs=(fs-mean)/std; xt=(core.features(tx)-mean)/std; y=torch.as_tensor(source['y'],device=device)
        core.seed_all(seed+100); m=models.Representation().cuda(); domain=nn.Sequential(nn.Linear(48,64),nn.SiLU(),nn.Linear(64,1)).cuda()
        # Gradient reversal changes the feature gradient sign, not the discriminator gradient.
        probe=torch.tensor([1.],device=device,requires_grad=True); core.reverse(probe,.5).sum().backward(); assert probe.grad.item()==-.5
        params=list(m.encoder.parameters())+list(m.classifier.parameters())+list(domain.parameters())
        opt=torch.optim.AdamW(params,lr=.001,weight_decay=.0001); g=torch.Generator(device=device).manual_seed(seed+101)
        steps=3 if a.check else 4500; history=[]
        for step in range(1,steps+1):
            if step==3001: opt=torch.optim.AdamW(params,lr=.0002,weight_decay=.0001)
            si=base.balanced(y,g); ti=base.sample(len(xt),len(si),device,g); n=len(si)
            m.train(); z=m.encode(torch.cat([xs[si],xt[ti]])); ce=F.cross_entropy(m.classifier(z[:n]),y[si])
            labels=torch.cat([torch.ones(n,device=device),torch.zeros(n,device=device)])
            dl=F.binary_cross_entropy_with_logits(domain(core.reverse(z,min(step/500,1.))).flatten(),labels)
            loss=ce+.2*dl; assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params,5); opt.step()
            if step%300==0 or step==steps:
                row={'step':step,'source_ce':float(ce.detach()),'domain_bce':float(dl.detach())}; history.append(row); print(row,flush=True)
        m.eval(); torch.save({'state_dict':m.state_dict(),'domain':domain.state_dict(),'mean':mean.cpu(),'std':std.cpu(),'step':steps},out/'dann_last.pt')
        copy=models.Representation().cuda(); copy.load_state_dict(torch.load(out/'dann_last.pt',weights_only=False)['state_dict']); copy.eval()
        with torch.no_grad():
            assert torch.equal(m.encode(xs[:36]),copy.encode(xs[:36]))
            sf=models.encode(m,xs).cpu().numpy(); tf=models.encode(m,xt).cpu().numpy()
        assert np.isfinite(sf).all() and np.isfinite(tf).all()
        np.savez_compressed(out/'dann_features.npz',source=sf,target_adapt=tf,source_y=source['y'],source_record_id=source['record_id'],target_record_id=target['record_id'])
        dump(out/'history.json',history); dump(out/'TRAINING_COMPLETE.json',{'target_test_read':False,'seconds':time.perf_counter()-start,'reload_verified':True})
        if not a.check:
            test=checked('target_test'); raw=torch.as_tensor(test['x'],device=device)
            g=torch.Generator(device=device).manual_seed(12345+'SML'.index(pair[-1]))
            result={}; saved={'y':test['y'],'record_id':test['record_id']}
            for condition,snr in core.LEVELS:
                x=(core.features(raw if snr is None else core.noisy(raw,snr,g))-mean)/std
                with torch.no_grad(): prob=torch.cat([m.classifier(m.encode(b)).softmax(-1) for b in x.split(256)]).cpu().numpy()
                result[condition]=evaluate(out,'dann',test,prob.argmax(1),condition); saved[condition]=prob
            np.savez_compressed(out/'dann_probabilities.npz',**saved)
            dump(out/'summary.json',{'dann':result})
    dump(out/'DONE.json',{'success':True,'check':a.check,'seconds':time.perf_counter()-start}); print('DONE',out,flush=True)

if __name__=='__main__': main()
