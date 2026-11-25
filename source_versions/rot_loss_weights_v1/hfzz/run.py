"""Frozen single-source UDA: explicit factors, cluster objective, style diffusion."""
from pathlib import Path
import argparse,copy,hashlib,json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from core import (ARMS,LEVELS,Classifier,FactorModel,StyleDenoiser,head,reverse,seed_all,features,noisy,
                  supcon,cross_covariance,local_geometry,chart,covariance,schedule,forward_noise,edit)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    temp=Path(str(p)+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(p)
def load(p,device):
    with np.load(p) as z:return {k:torch.as_tensor(z[k],device=device,dtype=torch.float32 if k=='x' else torch.long) for k in z.files}
def short(d):
    if 'y' in d:idx=torch.cat([torch.where(d['y']==c)[0][:4] for c in range(9)])
    else:idx=torch.arange(min(36,len(d['x'])),device=d['x'].device)
    return {k:v[idx] for k,v in d.items()}
def encode(model,x):
    with torch.no_grad():
        parts=[model.encode(a) for a in x.split(256)]
    return tuple(torch.cat([p[i] for p in parts]) for i in range(2))
def geometry(model,f,u,mean,std):
    parts=[local_geometry(model,f[i:i+128],u[i:i+128],mean,std) for i in range(0,len(u),128)]
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}
def take(d,idx):return {k:v[idx] for k,v in d.items()}
def sample(n,b,device,g):return torch.randint(n,(b,),device=device,generator=g)
def balanced(y,g):
    result=[]
    for c in range(9):
        ix=torch.where(y==c)[0];result.append(ix[sample(len(ix),4,y.device,g)])
    return torch.cat(result)

def train_factors(xs,ys,xt,args,cluster,out):
    seed_all(args.seed+100);model=FactorModel().to(xs.device)
    initial=copy.deepcopy(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    g=torch.Generator(device=xs.device).manual_seed(args.seed+101)
    prototypes=torch.zeros(9,32,device=xs.device);seen=torch.zeros(9,device=xs.device,dtype=torch.bool)
    history=[];start=time.time();name='full' if cluster else 'no_cluster'
    for step in range(1,args.factor_steps+1):
        model.train();si=balanced(ys,g);ti=sample(len(xt),len(si),xt.device,g);n=len(si)
        x=torch.cat([xs[si],xt[ti]]);d=torch.cat([torch.zeros(n),torch.ones(n)]).long().to(x.device)
        logits,f,s=model(x);ramp=min(step/500,1.)
        ce=F.cross_entropy(logits[:n],ys[si]);rec=F.mse_loss(model.decode(f,s),x)
        sd=F.cross_entropy(model.style_domain(s),d)
        fd=F.cross_entropy(model.fault_domain(reverse(f,ramp)),d)
        sf=F.cross_entropy(model.style_fault(reverse(s[:n],ramp)),ys[si])
        cov=cross_covariance(f,s)
        swapped=model.decode(f[:n],s[n:]);cf,cs=model.encode(swapped)
        cycle=F.mse_loss(cf,f[:n].detach())+F.mse_loss(cs,s[n:].detach())
        with torch.no_grad():
            for c in range(9):
                v=F.normalize(f[:n][ys[si]==c].mean(0),dim=0)
                prototypes[c]=F.normalize(.95*prototypes[c]+.05*v,dim=0) if seen[c] else v
                seen[c]=True
            prob=logits[n:].softmax(-1);confidence,pseudo=prob.max(-1)
            nearest=(f[n:]@prototypes.T).argmax(-1)
            accepted=(confidence>=.9)&(nearest==pseudo)&(step>500)
        cl=logits.sum()*0
        if cluster:
            cl=supcon(torch.cat([f[:n],f[n:][accepted]]),torch.cat([ys[si],pseudo[accepted]]))
        loss=ce+rec+.2*sd+.2*fd+.1*sf+.05*cov+.1*cycle+(.2*cl if cluster else 0)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite factor objective')
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%200==0 or step==args.factor_steps:
            row={'step':step,'class_ce':float(ce.detach()),'reconstruction_mse':float(rec.detach()),
                'style_domain_ce':float(sd.detach()),'fault_domain_adversary_ce':float(fd.detach()),
                'style_class_adversary_ce':float(sf.detach()),'cross_covariance':float(cov.detach()),
                'swap_cycle':float(cycle.detach()),'cluster_loss':float(cl.detach()),
                'target_pseudo_accepted_fraction':float(accepted.float().mean()),'seconds':time.time()-start}
            history.append(row);print(json.dumps({'phase':'factor_'+name,**row}),flush=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    torch.save({'state_dict':model.state_dict(),'step':args.factor_steps,'initial_state_dict':initial,'prototypes':prototypes},out/f'factor_{name}_last.pt')
    dump(out/f'factor_{name}_training.json',history)
    return model,prototypes

def train_diffusion(f,u,domains,geo,rot,mode,args,ab,out):
    seed_all(args.seed+200);net=StyleDenoiser().to(u.device);opt=torch.optim.AdamW(net.parameters(),lr=2e-4)
    g=torch.Generator(device=u.device).manual_seed(args.seed+201)
    src=torch.where(domains==0)[0];tgt=torch.where(domains==1)[0];history=[];start=time.time()
    for step in range(1,args.diffusion_steps+1):
        ix=torch.cat([src[sample(len(src),32,u.device,g)],tgt[sample(len(tgt),32,u.device,g)]])
        t=torch.randint(200,(len(ix),),device=u.device,generator=g)
        white=torch.randn(u[ix].shape,device=u.device,generator=g);c=chart(take(geo,ix),mode,rot)
        z=forward_noise(u[ix],t,white,c,ab);loss=F.mse_loss(net(z,t,f[ix],domains[ix],c),white)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite style diffusion')
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if step%300==0 or step==args.diffusion_steps:
            row={'step':step,'white_noise_mse':float(loss.detach()),'seconds':time.time()-start};history.append(row)
            print(json.dumps({'phase':'diffusion_'+mode,**row}),flush=True)
    net.eval();torch.save({'state_dict':net.state_dict(),'step':args.diffusion_steps,'mode':mode},out/f'diffusion_{mode}_last.pt')
    dump(out/f'diffusion_{mode}_training.json',history);return net

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
    pred=p.argmax(1);cm=np.zeros((9,9),int);np.add.at(cm,(y,pred),1)
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
        rp=np.zeros((len(ids),9));np.add.at(rp,ix,p);rp/=np.bincount(ix)[:,None]
        result[name]={'window':metric(y,p),'record':metric(ry,rp)}
    result['mean_accuracy']=float(np.mean([result[n]['window']['accuracy'] for n,_ in LEVELS]))
    np.savez_compressed(path,**saved);return result

def probe_geometry(model,f,u,geo,mean,std,rot,args):
    ix=torch.linspace(0,len(u)-1,min(256,len(u)),device=u.device).long()
    g=torch.Generator(device=u.device).manual_seed(args.seed+499)
    white=torch.randn(u[ix].shape,device=u.device,generator=g);result={}
    with torch.no_grad():
        p=model(model.decode(f[ix],u[ix]*std+mean))[0].softmax(-1)
        for mode in ['isotropic','foliated','random']:
            c=chart(take(geo,ix),mode,rot);delta=covariance(white,c)
            delta=.1*delta/delta.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
            q=model(model.decode(f[ix],(u[ix]+delta)*std+mean))[0].softmax(-1)
            kl=(p*(p.clamp_min(1e-9).log()-q.clamp_min(1e-9).log())).sum(-1).mean()
            result[mode]={'decoded_reencoded_class_KL':float(kl),'equal_standardized_style_RMS':.1}
    return {'scope':'source_and_unlabeled_target_training_references_only','modes':result,
            'rank_min':int(geo['rank'].min()),'rank_max':int(geo['rank'].max()),'interpretation':'Local posterior geometry only; not proof of physical disentanglement.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--factor-steps',type=int,default=3000);p.add_argument('--diffusion-steps',type=int,default=1500)
    p.add_argument('--classifier-steps',type=int,default=2400);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=False);out=args.out;device=torch.device(args.device)
    manifest=json.loads((args.data/'manifest.json').read_text());loaded={}
    for name in ['source_train','source_val','target_unlabeled']:
        assert sha(args.data/(name+'.npz'))==manifest['splits'][name]['sha256']
        loaded[name]=load(args.data/(name+'.npz'),device)
    train,val,adapt=[loaded[n] for n in ['source_train','source_val','target_unlabeled']]
    assert 'y' not in adapt
    assert set(train['rpm'].tolist())==set(val['rpm'].tolist())=={manifest['source_rpm']}
    assert set(adapt['rpm'].tolist())=={manifest['target_rpm']}
    for i,a in enumerate(loaded.values()):
        for b in list(loaded.values())[:i]:assert set(a['record_id'].tolist()).isdisjoint(b['record_id'].tolist())
    if args.check_only:train=short(train);val=short(val);adapt=short(adapt)
    f=features(train['x']);mean=f.mean((0,2),keepdim=True);std=f.std((0,2),keepdim=True).clamp_min(1e-6)
    xs=(f-mean)/std;xt=(features(adapt['x'])-mean)/std;y=train['y']
    va=arrays(val,mean,std,manifest['source_rpm']+12345)
    config={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'manifest':manifest,
        'normalization_fit':'source_train_only','target_adaptation_class_labels_available':False,'target_test_accessed':False,
        'source_sha256':{f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},'arms':ARMS,
        'normalization':{'mean':mean.cpu().tolist(),'std':std.cpu().tolist()},'selection':'fixed_last_step',
        'historical_exposure':True,'same_bearing_ids_across_speed':True,'method':'Explicit fault/style factors; supervised and confident-pseudo clustering; conditional target-style diffusion with local fixed posterior chart.'}
    dump(out/'run.json',config)
    full,prototypes=train_factors(xs,y,xt,args,True,out)
    nocluster,_=train_factors(xs,y,xt,args,False,out)
    fs,ss=encode(full,xs);ft,st=encode(full,xt);nfs,nss=encode(nocluster,xs);nft,nst=encode(nocluster,xt)
    with torch.no_grad():
        prob=full.logits(ft).softmax(-1);conf,pseudo=prob.max(1);nearest=(ft@prototypes.T).argmax(1)
        g=torch.Generator(device=device).manual_seed(args.seed+301);pair=[];fallback=0
        for label in y:
            possible=torch.where((pseudo==label)&(conf>=.9)&(nearest==pseudo))[0]
            if not len(possible):possible=torch.arange(len(xt),device=device);fallback+=1
            pair.append(possible[sample(len(possible),1,device,g)])
        pair=torch.cat(pair)
    dump(out/'target_style_pairing.json',{'count':len(pair),'fallback_global_target_count':fallback,
        'target_class_label_access':False,'target_predicted_counts':torch.bincount(pseudo,minlength=9).tolist(),
        'confident_target_fraction':float(((conf>=.9)&(nearest==pseudo)).float().mean()),
        'pair_indices_sha256':hashlib.sha256(pair.cpu().numpy().tobytes()).hexdigest()})
    sf=torch.cat([fs,ft]);styles=torch.cat([ss,st]);sm=styles.mean(0);sd=styles.std(0).clamp_min(.05);u=(styles-sm)/sd
    domains=torch.cat([torch.zeros(len(xs)),torch.ones(len(xt))]).long().to(device)
    geo=geometry(full,sf,u,sm,sd)
    gen=torch.Generator(device=device).manual_seed(args.seed+302);rot,_=torch.linalg.qr(torch.randn(16,16,device=device,generator=gen))
    white=torch.randn((len(xs),16),device=device,generator=gen);ab=schedule(device)
    torch.save({'mean':sm,'std':sd,'rotation':rot,'source_references':u[:len(xs)],'source_fault':fs,'source_geometry':take(geo,slice(0,len(xs)))},out/'style_chart.pt')
    dump(out/'geometry_probe.json',probe_geometry(full,sf,u,geo,sm,sd,rot,args))
    summary={}
    for arm in ARMS:
        print(json.dumps({'phase':'arm_start','arm':arm}),flush=True);pool=None
        with torch.no_grad():
            if arm=='reconstruction':pool=full.decode(fs,ss)
            elif arm=='swap_no_cluster':pool=nocluster.decode(nfs,nst[pair])
            elif arm=='swap_full':pool=full.decode(fs,st[pair])
        if arm.endswith('_latent_diffusion'):
            mode=arm.removesuffix('_latent_diffusion');net=train_diffusion(sf,u,domains,geo,rot,mode,args,ab,out)
            parts=[]
            for i in range(0,len(xs),128):
                sl=slice(i,min(i+128,len(xs)));c=chart(take(geo,sl),mode,rot)
                edited=edit(net,u[:len(xs)][sl],fs[sl],c,white[sl],ab,steps=2 if args.check_only else 16)
                with torch.no_grad():parts.append(full.decode(fs[sl],edited*sd+sm))
            pool=torch.cat(parts);del net
        diagnostics={}
        if pool is not None:
            assert pool.shape==xs.shape and torch.isfinite(pool).all()
            with torch.no_grad():
                pf,ps=encode(full,pool)
                diagnostics={'source_internal_class_agreement':float((full.logits(pf).argmax(1)==y).float().mean()),
                    'target_style_domain_probability':float(full.style_domain(ps).softmax(-1)[:,1].mean()),
                    'reencoded_fault_cosine':float((fs*pf).sum(1).mean()),'edit_rms':float((pool-xs).square().mean().sqrt()),
                    'interpretation':'Internal consistency only; not guaranteed true labels or physically correct target speed.'}
            np.savez_compressed(out/f'{arm}_pool.npz',x=pool.cpu().numpy(),y=y.cpu().numpy(),source_record_id=train['record_id'].cpu().numpy())
        model=train_classifier(arm,xs,y,xt,pool,args,out,mean,std)
        summary[arm]={'source_validation':evaluate(model,va,val,out/f'{arm}_last_source_val_predictions.npz'),
                      'classifier_final_step':args.classifier_steps,'generation':diagnostics}
        dump(out/'source_summary.json',summary);del model,pool
    dump(out/'TRAINING_COMPLETE.json',{'arms':ARMS,'step':args.classifier_steps,'target_test_accessed':False})
    audit={}
    for arm in ARMS:
        ck=torch.load(out/f'{arm}_last.pt',map_location=device,weights_only=False);assert ck['step']==args.classifier_steps
        model=Classifier().to(device);model.load_state_dict(ck['state_dict']);model.eval()
        path=out/f'{arm}_recomputed_source_val.npz';evaluate(model,va,val,path)
        with np.load(path) as a,np.load(out/f'{arm}_last_source_val_predictions.npz') as b:
            error=max(float(np.max(np.abs(a[n]-b[n]))) for n,_ in LEVELS)
        assert error<1e-7; audit[arm]={'max_probability_error':error,'weight_sha256':sha(out/f'{arm}_last.pt')}
    dump(out/'weight_audit.json',audit)
    if not args.check_only:
        assert sha(args.data/'target_test.npz')==manifest['splits']['target_test']['sha256']
        target=load(args.data/'target_test.npz',device)
        for old in [train,val,adapt]:assert set(target['record_id'].tolist()).isdisjoint(old['record_id'].tolist())
        assert set(target['rpm'].tolist())=={manifest['target_rpm']}
        ta=arrays(target,mean,std,manifest['target_rpm']+12345)
        dump(out/'TARGET_EVALUATION_STARTED.json',{'after_all_training_and_reload':True,'run_sha256':sha(out/'run.json')})
        for arm in ARMS:
            ck=torch.load(out/f'{arm}_last.pt',map_location=device,weights_only=False)
            model=Classifier().to(device);model.load_state_dict(ck['state_dict']);model.eval()
            summary[arm]['target']=evaluate(model,ta,target,out/f'{arm}_last_target_predictions.npz')
        from diagnostics import diagnose
        diagnose(full,nocluster,xs,xt,va['clean'],ta['clean'],train,adapt,val,target,out,device)
        from plot_clusters import plot
        plot(out)
    dump(out/'summary.json',summary)
    dump(out/'DONE.json',{'success':True,'target_evaluated':not args.check_only,'arms':ARMS,'classifier_final_step':args.classifier_steps})
    print(json.dumps({'DONE':True,'target_clean':{k:v.get('target',{}).get('clean',{}).get('window',{}).get('accuracy') for k,v in summary.items()}}),flush=True)
if __name__=='__main__':main()
