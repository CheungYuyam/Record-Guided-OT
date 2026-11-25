"""Record-disjoint HFZZ windows; no concatenated summary files are used."""
from pathlib import Path
import json,re,hashlib,argparse
import numpy as np
from scipy.io import loadmat

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(raw, out, ntrain=3, target=None):
    out.mkdir(parents=True,exist_ok=True)
    windows=[]; labels=[]; speeds=[]; ids=[]; metadata=[]; split=[]
    bearings=['B01','B02','B03','B04','B06','B07','B08','B09','B14']
    rng=np.random.default_rng(20261010)
    for folder in sorted(raw.iterdir()):
        match=re.search(r'【(\d+)】(B\d+)_12.8k_(\d+)RPM',folder.name)
        if not match: continue
        group,bearing,rpm=match.groups(); label=bearings.index(bearing); rpm=int(rpm)
        files=[]
        for p in folder.glob('*.mat'):
            m=re.search(r'_(\d+)\.mat$',p.name)
            if m and int(m[1])>=2: files.append((int(m[1]),p))
        files.sort(); assert len(files)==30,(folder,len(files))
        perm=rng.permutation(30)
        roles={int(i):('train' if j<ntrain else 'val' if j<ntrain+5 else 'test') for j,i in enumerate(perm)}
        for index,(suffix,p) in enumerate(files):
            x=np.asarray(loadmat(p)['Datas'],dtype=np.float32)
            assert x.ndim==2 and x.shape[1]==2 and len(x)>=16384 and np.isfinite(x).all()
            rid=len(metadata); role=roles[index] if target!=rpm else 'test'
            metadata.append(dict(record_id=rid,group=group,bearing=bearing,label=label,rpm=rpm,
                path=str(p.relative_to(raw)),sha256=sha(p),original_points=len(x),retained_points=16384,split=role))
            # fixed 16 non-overlapping windows; extra tail recorded, not silently reweighted
            w=x[:16384].reshape(16,1024,2).transpose(0,2,1).copy()
            windows.append(w); labels.extend([label]*16); speeds.extend([rpm]*16);ids.extend([rid]*16);split.extend([role]*16)
    x=np.concatenate(windows); y=np.array(labels); rpm=np.array(speeds); rid=np.array(ids); split=np.array(split)
    assert len(metadata)==810 and x.shape==(12960,2,1024)
    sets={name:set(rid[split==name].tolist()) for name in ('train','val','test')}
    assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
    for name in sets:
        keep=split==name
        np.savez_compressed(out/f'{name}.npz',x=x[keep],y=y[keep],rpm=rpm[keep],record_id=rid[keep])
    manifest=dict(split_seed=20261010,ntrain_per_group=ntrain,target_speed=target,sampling_hz=12800,window=1024,
        labels=bearings,records=metadata,counts={name:dict(records=len(sets[name]),windows=int((split==name).sum())) for name in sets},
        source_data_hash=hashlib.sha256(''.join(m['sha256'] for m in metadata).encode()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(counts=manifest['counts'],anomalies=[m for m in metadata if m['original_points']!=16384]),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ntrain',type=int,default=3);p.add_argument('--target',type=int)
    a=p.parse_args();prepare(a.raw,a.out,a.ntrain,a.target)
