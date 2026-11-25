"""Six directed single-source UDA tasks. Adaptation files contain no class labels."""
from pathlib import Path
import argparse,hashlib,json,itertools
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False)
    files=[a.input/n for n in ['train.npz','val.npz','test.npz']]
    z=[np.load(n) for n in files];data={k:np.concatenate([v[k] for v in z]) for k in z[0].files}
    records={int(i):(int(data['y'][data['record_id']==i][0]),int(data['rpm'][data['record_id']==i][0])) for i in np.unique(data['record_id'])}
    assert len(records)==810
    groups={}
    for rpm,c in itertools.product([1750,2000,2250],range(9)):
        ids=sorted(i for i,(yy,rr) in records.items() if yy==c and rr==rpm);assert len(ids)==30
        groups[(rpm,c)]=np.random.default_rng(20261020+rpm*10+c).permutation(ids).tolist()
    allinfo={'protocol':'single_source_UDA_with_record_disjoint_target_adaptation_and_test',
        'input_sha256':{f.name:sha(f) for f in files},'split_seed':20261020,'tasks':{},
        'historical_exposure':True,'same_bearing_ids_across_speeds':True,
        'label_access':'Labels used to stratify split; target adaptation npz has no y; target test opened only after all classifier training.'}
    for source,target in itertools.permutations([1750,2000,2250],2):
        name=f'{source}_to_{target}';folder=a.out/name;folder.mkdir()
        selection={'source_train':sum([groups[(source,c)][:25] for c in range(9)],[]),
            'source_val':sum([groups[(source,c)][25:] for c in range(9)],[]),
            'target_unlabeled':sum([groups[(target,c)][:15] for c in range(9)],[]),
            'target_test':sum([groups[(target,c)][15:] for c in range(9)],[])}
        ss=[set(v) for v in selection.values()];assert all(a.isdisjoint(b) for a,b in itertools.combinations(ss,2))
        info={'source_rpm':source,'target_rpm':target,'splits':{}}
        for split,ids in selection.items():
            mask=np.isin(data['record_id'],ids);path=folder/(split+'.npz')
            payload={k:v[mask] for k,v in data.items() if not (split=='target_unlabeled' and k=='y')}
            np.savez_compressed(path,**payload)
            info['splits'][split]={'record_ids':sorted(ids),'records':len(ids),'windows':int(mask.sum()),'sha256':sha(path),'contains_y':'y' in payload}
        (folder/'manifest.json').write_text(json.dumps(info,indent=2))
        allinfo['tasks'][name]=info
    (a.out/'manifest.json').write_text(json.dumps(allinfo,indent=2))
    print(json.dumps({'prepared':str(a.out),'tasks':6,'source_train_windows':3600,'target_adaptation_windows':2160,'target_test_windows':2160}))
if __name__=='__main__':main()
