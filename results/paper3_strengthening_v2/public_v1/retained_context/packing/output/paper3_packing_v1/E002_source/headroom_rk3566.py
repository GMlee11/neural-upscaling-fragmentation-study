"""E002 bounded DEVELOPMENT screen; not confirmatory evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from qualify_rk3566 import BASE,PIN,Factory,verify,collector,BackendWorker,execute_service,resize_bicubic,extract_halo_patch,extract_zero_halo_patch,make_dispatch_descriptor,SHAPES
from packing.core import pack,grouped_rows
from packing.service import execute
from packing.jobs_service import execute_jobs
from src.experiments.paper3_dispatch_successor_v1.core import development_masks,plan
from src.experiments.paper3_dispatch_successor_v1.development import output_integrity

def exact_residual(image,tile,b):
    r,c=divmod(tile,10);h=b.halo
    d=make_dispatch_descriptor(frame_id=0,frame_width=640,frame_height=360,core_x=c*64,core_y=r*60,core_width=64,core_height=60,halo=h,deadline_us=33333,priority=1.)
    if b.boundary=='zero':p,x,y=extract_zero_halo_patch(image,d,halo=h)
    else:p=extract_halo_patch(image,d,halo=h);x=y=h
    raw=b.infer((64,60),p)
    return np.ascontiguousarray(raw[0,:,y:y+60,x:x+64])

def sample(req,b):
    rows=req['rows'];policy=req['policy']
    if policy=='reference':
        out,m=execute_service(b.fixture,rows,'always_exact',b,b.halo,{'boundary_mode':b.boundary})
        b.screen_reference=out;b.screen_mask=rows
        b.runtime.validate_outputs()
        return out,{'reference_only':True,'passed':True}
    if rows!=b.screen_mask:raise ValueError('missing matching reference')
    if policy in ('exact','row_spans','grouped'):
        planner=grouped_rows if policy=='grouped' else lambda r:plan(r,policy)
        out,m=execute_jobs(b.fixture,rows,planner,b)
    else:
        w,h=req['shape']
        out,m=execute(b.fixture,rows,lambda r:pack(r,b.halo,(w+2*b.halo,h+2*b.halo)),resize_bicubic,
            lambda canvas,value:b.infer((w,h),value),lambda im,tile,halo:exact_residual(im,tile,b))
    # All backend finite-output/FD validation is outside timed service equally.
    witness=b.runtime.validate_outputs()
    try:integrity=output_integrity(out,b.screen_reference,b.baseline,rows);passed=True
    except ValueError as error:integrity={'failure':str(error)};passed=False
    return out,dict(m,passed=passed,integrity=integrity,fd_witness=witness)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cell',choices=('width24','fp16'),required=True);p.add_argument('--source-manifest-sha',required=True);a=p.parse_args()
    here=Path(__file__).resolve().parent;raw=(here/'source_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.source_manifest_sha:raise ValueError('manifest mismatch')
    for rel,pin in json.loads(raw)['files'].items():
        if hashlib.sha256((here/rel).read_bytes()).hexdigest()!=pin:raise ValueError('source drift')
    for cell in ('width24','fp16'):
        old=json.loads((here.parent/('E001_'+cell)/'journal.jsonl').read_text().splitlines()[-1])
        if old['state']!='complete':raise ValueError('E001 not complete')
    bundle,pins,_=verify(BASE,PIN,a.cell,'engineering',0)
    outdir=here.parent/('E002_'+a.cell);outdir.mkdir(exist_ok=False)
    masks=[m for m in development_masks() if m['selected_tiles'] in (12,36) and m['mask_id'].endswith('_0')]
    policies=[{'policy':x} for x in ('exact','row_spans','grouped')]+[{'policy':'pack_'+str(w)+'x'+str(h),'shape':[w,h]} for w,h in SHAPES]
    worker=None;owned={os.getpid()}
    with (outdir/'journal.jsonl').open('x',buffering=1) as log:
        def emit(v):log.write(json.dumps(dict(v,unix_time=time.time()),sort_keys=True,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
        def control(stage):
            s=collector.collect(BASE,owned);emit({'state':'control','stage':stage,'snapshot':s});collector.validate(s)
            if s['invariants']!=pins['qualified_control_invariants']:raise ValueError('changed invariants')
            return s
        def retain(value):
            sha=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest();dest=outdir/('raw_'+sha+'.npz')
            if not dest.exists():np.savez_compressed(dest,output=value)
            return {'path':dest.name,'array_sha256':sha,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        try:
            emit({'state':'running','cell':a.cell,'source_manifest_sha':a.source_manifest_sha,'purpose':'headroom_development_only','masks':masks})
            if control('start')['temperature_c']>60:raise ValueError('above60C start')
            t=time.perf_counter();worker=BackendWorker(Factory(a.cell,bundle['graph_mapping'][a.cell]));owned.add(worker.process.pid)
            emit({'state':'backend_loaded','initialization_warmup_s':time.perf_counter()-t});control('loaded')
            index=0
            for k,mask in enumerate(masks):
                req=dict(mask,policy='reference');value,m=worker.sample(sample,req)
                emit({'state':'reference','request':req,'metrics':m,**retain(value)})
                for policy in policies[k:]+policies[:k]:
                    control('before_'+str(index));req=dict(mask,**policy);value,m=worker.sample(sample,req)
                    emit({'state':'output_retained','index':index,'request':req,'metrics':m,**retain(value)})
                    control('after_'+str(index))
                    if not m['passed']:raise ValueError('numerical gate failed')
                    index+=1
            worker.close();worker=None;owned={os.getpid()};control('closed');emit({'state':'complete','observations':index,'confirmation':False})
        except BaseException as error:emit({'state':'failed','error':repr(error)});raise
        finally:
            if worker is not None:worker.close()

if __name__=='__main__':main()
