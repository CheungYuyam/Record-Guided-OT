"""Explicit-path adapter for the archived data preparation scripts. No training."""
from pathlib import Path
import argparse, importlib.util, subprocess, sys

HERE=Path(__file__).resolve().parent
def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/'original_scripts'/filename)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='dataset',required=True)
    hf=sub.add_parser('hfzz');hf.add_argument('--raw',type=Path,required=True);hf.add_argument('--pool',type=Path,required=True);hf.add_argument('--out',type=Path,required=True)
    pt=sub.add_parser('pt700');pt.add_argument('--archive',type=Path,required=True);pt.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();assert not a.out.exists(),f'Output already exists: {a.out}'
    if a.dataset=='hfzz':
        assert a.raw.is_dir();assert not a.pool.exists(),f'Pool already exists: {a.pool}'
        module('hfzz_raw','prepare_hfzz_raw.py').prepare(a.raw,a.pool,ntrain=3,target=None)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run([sys.executable,'-B',str(HERE/'original_scripts/prepare_hfzz_transfers.py'),'--input',str(a.pool),'--out',str(a.out)],check=True)
    else:
        assert a.archive.is_file()
        mod=module('pt700_raw','prepare_pt700_archive.py')
        mod.ARCHIVE=a.archive.resolve();mod.OUT=a.out.resolve();mod.main()

if __name__=='__main__':main()
