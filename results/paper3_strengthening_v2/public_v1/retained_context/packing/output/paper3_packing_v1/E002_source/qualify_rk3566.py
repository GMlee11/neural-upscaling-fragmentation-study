"""E001 engineering numerical action; no performance evaluation or model fitting.

Uses the closed transfer bundle read-only and writes only a fresh sibling
engineering action. New packing modules live alongside this script, never in
the sealed bundle. Native work has an owned subprocess watchdog.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


BASE=Path('/home/paper3/experiments/paper3-transfer-v1/bundle')
PIN='74d309d732f787b30da8e3488a9877963963f58b94b9f7a81a0d5591b8143ce0'
sys.path.insert(0,str(BASE))
from src.experiments.paper3_transfer_v1.target_entrypoint import verify
from src.experiments.paper3_dispatch_rk3566_v1.numerical import NumericalBackend
from src.experiments.paper3_dispatch_rk3566_v1 import collector
from src.experiments.paper3_dispatch_successor_v1.worker import BackendWorker
from src.experiments.paper3_dispatch_successor_v1.development import execute_service
from src.deployment.benchmark_rk3576_sparse import make_dispatch_descriptor, extract_halo_patch, resize_bicubic
from src.deployment.paper3_progressive_zero_padding_v2 import extract_zero_halo_patch
from packing.core import pack, compose
from packing.service import execute

ROWS=('1111111111',)*6  # Open engineering geometry, not reserved evaluation.
SHAPES=((64,60),(128,60),(256,60),(512,60),(576,60),(640,60),(640,120),(640,180))


@dataclass
class Factory:
    cell:str
    mapping:dict
    def __call__(self): return NumericalBackend(BASE,self.cell,self.mapping)


def exact(image,tile,backend):
    r,c=divmod(tile,10); h=backend.halo
    desc=make_dispatch_descriptor(frame_id=0,frame_width=640,frame_height=360,
        core_x=c*64,core_y=r*60,core_width=64,core_height=60,halo=h,deadline_us=33333,priority=1.)
    if backend.boundary=='zero': value,x,y=extract_zero_halo_patch(image,desc,halo=h)
    else: value=extract_halo_patch(image,desc,halo=h); x=y=h
    raw=backend.infer((64,60),value)
    if raw.shape!=(1,12,value.shape[0],value.shape[1]): raise ValueError('raw residual layout')
    result=np.ascontiguousarray(raw[0,:,y:y+60,x:x+64])
    backend.runtime.validate_outputs()
    return result


def sample(request,backend):
    image=backend.fixture; h=backend.halo
    if request['kind']=='reference':
        residuals={i:exact(image,i,backend) for i in range(60)}
        value=compose(resize_bicubic(image),pack(ROWS,h,(64+2*h,60+2*h)),residuals)
        original,_=execute_service(image,ROWS,'always_exact',backend,h,{'boundary_mode':backend.boundary})
        backend.runtime.validate_outputs()
        agreement=float(np.max(np.abs(value-original)))
        backend.packing_reference=original
        backend.packing_full_reference=np.load(BASE/'reference/full_reference.npy',allow_pickle=False) if backend.cell=='fp16' else original
        return value,{'kind':'reference','original_exact_max_error':agreement,'passed':agreement<=2e-6,
                      'rknpu_fd_witness':backend.runtime.validate_outputs()}
    w,hcore=request['shape']
    def infer(canvas,value):
        raw=backend.infer((w,hcore),value)
        backend.runtime.validate_outputs()
        return raw
    value,_=execute(image,ROWS,lambda rows:pack(rows,h,(w+2*h,hcore+2*h),reverse=request['reverse']),
        resize_bicubic,infer,lambda image,tile,halo:exact(image,tile,backend))
    reference=backend.packing_reference
    mae=float(np.mean(np.abs(value.astype(np.float64)-reference)))
    full=backend.packing_full_reference.astype(np.float64)
    ref_mae=float(np.mean(np.abs(value.astype(np.float64)-full)))
    ref_mse=float(np.mean((reference.astype(np.float64)-full)**2))
    mse=float(np.mean((value.astype(np.float64)-full)**2))
    # Width24 uses accepted same-backend exact as reference, so PSNR change to
    # a full-frame FP reference is not asserted by this engineering action.
    delta=0. if ref_mse==mse else abs(10*np.log10(ref_mse/mse)) if ref_mse>0 and mse>0 else None
    passed=mae<=.001 and (backend.cell!='fp16' or (ref_mae<=.001 and delta is not None and delta<=.02))
    return value,dict(request,cross_exact_mae=mae,reference_mae=ref_mae,
        fp16_psnr_delta_db=delta if backend.cell=='fp16' else None,
        passed=passed,rknpu_fd_witness=backend.runtime.validate_outputs())


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--cell',choices=('width24','fp16'),required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--source-manifest-sha',required=True)
    a=parser.parse_args(); here=Path(__file__).resolve().parent
    manifest=(here/'source_manifest.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest()!=a.source_manifest_sha: raise ValueError('source manifest mismatch')
    for rel,pin in json.loads(manifest)['files'].items():
        if hashlib.sha256((here/rel).read_bytes()).hexdigest()!=pin: raise ValueError('source changed')
    bundle,pins,_=verify(BASE,PIN,a.cell,'engineering',0)
    if a.output.resolve()!=here/f'E001_{a.cell}': raise ValueError('fresh exact engineering destination required')
    # The old launcher must be terminal; its outputs are already copied locally.
    previous=json.loads((BASE.parent/'evaluation_launcher/events.jsonl').read_text().splitlines()[-1])
    if previous['state']!='complete': raise ValueError('previous capture not terminal')
    a.output.mkdir(exist_ok=False)
    worker=None; owned={os.getpid()}
    with (a.output/'journal.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        def emit(event):
            log.write(json.dumps(dict(event,unix_time=time.time()),sort_keys=True,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
        def control(stage):
            value=collector.collect(BASE,owned);emit({'state':'control','stage':stage,'snapshot':value})
            collector.validate(value)
            if value['invariants']!=pins['qualified_control_invariants']: raise ValueError('accepted controls changed')
            return value
        try:
            emit({'state':'running','purpose':'numerical_engineering_only','cell':a.cell,'source_manifest_sha':a.source_manifest_sha,'parent_bundle_sha':PIN})
            value=control('before_backend')
            if value['temperature_c']>60: raise ValueError('engineering start above60C; no inference')
            worker=BackendWorker(Factory(a.cell,bundle['graph_mapping'][a.cell]));owned.add(worker.process.pid)
            control('backend_loaded')
            requests=[{'kind':'reference'}]+[{'kind':'packing','shape':list(s),'reverse':r} for s in SHAPES for r in (False,True)]
            for i,request in enumerate(requests):
                control('before_'+str(i))
                value,metrics=worker.sample(sample,request)
                path=a.output/f'output_{i:02d}.npz'
                np.savez_compressed(path,output=value)
                emit({'state':'output_retained','index':i,'request':request,'metrics':metrics,
                      'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                if not metrics['passed']: raise ValueError('numerical gate failed; raw output retained')
            worker.close();worker=None;owned={os.getpid()};control('after_close')
            emit({'state':'complete','outputs':len(requests),'timing_claim':False})
        except BaseException as error:
            emit({'state':'failed','error':repr(error)});raise
        finally:
            if worker is not None: worker.close()

if __name__=='__main__': main()
