from pathlib import Path
import json,csv,hashlib,itertools
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];confusions={};sources={};selections={};duplicate_audit={}
for index in range(18):
 d=ROOT/f'results/pt700_record_repeats_v1_{index:03d}'
 assert read(d/'DONE.json')=={'success':True,'check':False}
 run=read(d/'run.json');mf=run['manifest'];repeat=mf['repeat'];pair=mf['source_domain']+'_to_'+mf['target_domain'];data=ROOT/f'data/pt700_record_repeats_v1/repeat{repeat}/{pair}'
 assert mf==read(data/'manifest.json') and run['protocol']==read(HERE/'protocol.json')
 assert run['code_sha256']==sha(HERE/'experiment.py')
 assert not read(d/'TRAINING_COMPLETE.json')['target_test_read']
 window_hashes={};record_hashes={}
 for split in ['source_train','source_val','target_unlabeled','target_test']:
  with np.load(data/(split+'.npz')) as zz:
   x=zz['x'];rr=zz['record_id']
   window_hashes[split]={hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest() for w in x}
   record_hashes[split]={hashlib.sha256(np.ascontiguousarray(x[rr==r]).tobytes()).hexdigest() for r in np.unique(rr)}
 overlaps={f'{a}__{b}':{'exact_windows':len(window_hashes[a]&window_hashes[b]),'exact_records':len(record_hashes[a]&record_hashes[b])} for a,b in itertools.combinations(window_hashes,2)}
 duplicate_audit[str(index)]=overlaps
 assert all(v['exact_windows']==v['exact_records']==0 for v in overlaps.values()),(index,overlaps)
 selections[str(index)]=mf;summary=read(d/'summary.json')
 with np.load(data/'target_test.npz') as z:truth={k:z[k].copy() for k in ['y','record_id']}
 for arm,conditions in summary.items():
  for condition,units in conditions.items():
   f=d/f'{arm}_{condition}.npz';sources[f.relative_to(ROOT).as_posix()]=sha(f)
   with np.load(f) as z:
    y,rid,score=z['y'],z['record_id'],z['score'];assert np.array_equal(y,truth['y']) and np.array_equal(rid,truth['record_id'])
    assert len(y)==960 and np.isfinite(score).all()
    for unit,yy,ss in [('window',y,score),('record',np.array([y[rid==r][0] for r in np.unique(rid)]),np.stack([score[rid==r].mean(0) for r in np.unique(rid)]))]:
     cm=np.zeros((4,4),int);np.add.at(cm,(yy,ss.argmax(1)),1)
     acc=float(100*cm.trace()/cm.sum());f1=float(100*np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)))
     assert np.array_equal(cm,units[unit]['confusion']) and abs(acc-units[unit]['accuracy_percent'])<1e-9 and abs(f1-units[unit]['macro_f1_percent'])<1e-9
     rows.append(dict(dataset='pt700',pair=pair,repeat=repeat,split_seed=mf['split_seed'],seed=mf['train_seed'],arm=arm,split='target',condition=condition,unit=unit,accuracy_percent=acc,macro_f1_percent=f1))
     confusions[f'{pair}_repeat{repeat}_{arm}_{condition}_{unit}']=cm.tolist()
out=ROOT/'deployment/pt700_record_repeats_v1_summary';out.mkdir(exist_ok=True)
with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
report={'runs':18,'model_fits':126,'metrics':len(rows),'all_predictions_recomputed':True,'sources':sources,'all_manifests':selections,'confusions':confusions,'protocol':read(HERE/'protocol.json'),'frozen':read(ROOT/'data/pt700_record_repeats_v1/FROZEN.json'),'results':{}}
report['exact_waveform_cross_split_duplicate_audit']=duplicate_audit
report['normalization']='All methods use source-training per-channel mean and sample standard deviation after identical RMS/log-rFFT features; classical float64 vs neural float32 arithmetic.'
for arm in summary:
 vals=[];stats={}
 for pair in ['S_to_M','S_to_L','M_to_S','M_to_L','L_to_S','L_to_M']:
  v=[r['accuracy_percent'] for r in rows if r['arm']==arm and r['pair']==pair and r['condition']=='clean' and r['unit']=='window'];assert len(v)==3
  stats[pair]={'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1)),'values':v};vals+=v
 stats['overall_mean']=float(np.mean(vals));report['results'][arm]=stats
(out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['results'],indent=2))
