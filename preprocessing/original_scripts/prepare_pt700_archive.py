"""One-time signal audit and frozen temporal split preparation; no training."""
from pathlib import Path
import sys, json, hashlib, zipfile, io, re, itertools, collections
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'deployment/audit_dependencies'))
import numpy as np
from scipy.signal import resample_poly
ARCHIVE=Path('Archive.zip')
OUT=R/'data/pt700_v1'
def sha(x):return hashlib.sha256(x).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def decoded(n):
    try:return n.encode('cp437').decode('utf8')
    except UnicodeError:return n
def main():
    assert not OUT.exists(),'Do not overwrite prepared data'
    assert sha(ARCHIVE.read_bytes())=='61b2d1df7aabb1a5c43b50003125fa83b8afd9a076c7d0725f625d8da5887b24'
    OUT.mkdir(parents=True)
    records={};hashes=collections.defaultdict(list);manifest=[]
    slots=np.rint(np.linspace(0,24,16)).astype(int)
    assert len(set(slots))==16
    with zipfile.ZipFile(ARCHIVE) as z:
        entries=[]
        for info in z.infolist():
            n=decoded(info.filename)
            if '__MACOSX' in n:continue
            m=re.search(r'单一故障/([SML])/([SML])([BIOT])/(\d+)\.txt$',n)
            if m:
                domain,dd,cls,number=m.groups();assert domain==dd
                entries.append((domain,cls,int(number),info))
        assert len(entries)==600
        for domain,cls,num,info in sorted(entries,key=lambda a:a[:3]):
            data=z.read(info)
            x=np.loadtxt(io.BytesIO(data),delimiter=',',skiprows=1,usecols=2,dtype=np.float32)
            assert x.shape==(204800,) and np.isfinite(x).all() and x.std()>0
            key=f'{domain}/{domain}{cls}/{num}.txt';h=sha(x.astype('<f4').tobytes());hashes[h].append(key)
            row={'path':key,'signal_sha256':h,'byte_sha256':sha(data),'rows':len(x),'number':num,
                 'domain':domain,'class':cls,'unique_amplitudes':len(np.unique(x))}
            manifest.append(row)
            # A per-file anti-aliasing filter cannot cross train/test boundaries.
            if num<=45:
                down=resample_poly(x,1,8,window=('kaiser',5.0),padtype='constant').astype(np.float32)
                assert down.shape==(25600,)
                windows=down.reshape(25,1024)[slots,None,:]
                rid='SML'.index(domain)*1000+'BIOT'.index(cls)*100+num
                records[domain,cls,num]=(windows,rid,h)
            if len(manifest)%50==0:print({'audited':len(manifest)},flush=True)
    duplicates=[v for v in hashes.values() if len(v)>1]
    # Reject any unexpected duplicates instead of adjusting splits after outcomes.
    assert duplicates==[['L/LI/49.txt','L/LI/50.txt']],duplicates
    split_numbers={'source_train':list(range(1,26)),'source_val':list(range(31,41)),
                   'target_unlabeled':list(range(1,21)),'target_test':list(range(31,46))}
    for s,t in itertools.permutations('SML',2):
        folder=OUT/f'{s}_to_{t}';folder.mkdir()
        mf={'source_domain':s,'target_domain':t,'classes':list('BIOT'),'splits':{},
            'sampling_hz_original_declared':102400,'sampling_hz_processed':12800,
            'window_length':1024,'windows_per_record':16,'selected_window_slots':slots.tolist(),
            'domain_axis':'fault severity, confounded with specimen/acquisition',
            'independent_bearing_validation':False,'target_labels_for_training':False}
        used_signals=set();used_windows=set();used_ids=set()
        for split,nums in split_numbers.items():
            d=s if split.startswith('source') else t
            xx=[];yy=[];rr=[];paths=[]
            for c,cls in enumerate('BIOT'):
                for n in nums:
                    w,rid,h=records[d,cls,n]
                    assert h not in used_signals and rid not in used_ids
                    used_signals.add(h);used_ids.add(rid)
                    for window in w:
                        wh=sha(window.astype('<f4').tobytes());assert wh not in used_windows
                        used_windows.add(wh)
                    xx.append(w);yy.extend([c]*16);rr.extend([rid]*16);paths.append(f'{d}/{d}{cls}/{n}.txt')
            arrays={'x':np.concatenate(xx),'record_id':np.array(rr,dtype=np.int64)}
            if split!='target_unlabeled':arrays['y']=np.array(yy,dtype=np.int64)
            path=folder/(split+'.npz');np.savez_compressed(path,**arrays)
            mf['splits'][split]={'sha256':sha(path.read_bytes()),'windows':len(rr),
                'record_ids':sorted(set(rr)),'files':paths,'file_numbers':nums}
        dump(folder/'manifest.json',mf)
    dump(OUT/'signal_audit.json',{'all_fault_files':manifest,'signal_duplicate_groups':duplicates,
         'normal_excluded':True,'compound_missing':True,'all_six_directions_record_signal_window_disjoint':True,
         'source_archive_sha256':sha(ARCHIVE.read_bytes()),'numpy':np.__version__,
         'scipy':__import__('scipy').__version__})
    print({'complete':True,'output':str(OUT)},flush=True)
if __name__=='__main__':main()
