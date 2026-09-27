"""Append-only RK3576 strong-baseline DEVELOPMENT screen, no confirmation.

Models and old source bundle are read-only. This file and strength/ live in a
new sibling source overlay. Do not import or execute predecessor entrypoints.
"""
import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np

BASE=Path('/home/radxa/experiments/paper3-transfer-v1-qualified/bundle')
PIN='6985b7171d109228ff8ed55584b23b0122f045d80f147dbfbb3c1a81d2ec5abc'
sys.path.insert(0,str(BASE))
from src.experiments.paper3_transfer_v1.entrypoint import verify
from src.experiments.paper3_transfer_v1.controls import collect,invariant_projection
from src.experiments.paper3_dispatch_successor_v1 import qualify
from src.experiments.paper3_dispatch_successor_v1.core import Job,WIDTHS,plan,development_masks,digest
from src.experiments.paper3_dispatch_successor_v1.development import output_integrity
from src.experiments.paper3_dispatch_successor_v1.worker import BackendWorker
from strength.core import grouped_rows,partition,per_row,lookup_cost,affine_cost,row_class
from strength.service import execute
from strength import ownership


@dataclass
class Factory:
    cell:str
    def __call__(self):return qualify.NumericalBackend(self.cell)


def mask_from_job(job):
    ids=set(job.commits)
    return tuple(''.join('1' if r*10+c in ids else '0' for c in range(10)) for r in range(6))


def sample(req,b):
    kind=req['kind']
    if kind=='reference':
        rows=('1111111111',)*6
        out,m=execute(b.fixture,rows,lambda r:plan(r,'exact'),b,'whole')
        b.strong_reference=out
    else:
        rows=tuple(req['rows']);mode=req['reconstruction']
        if kind in ('qualification','profile'):
            j=req['job'];job=Job(j['x'],j['y'],j['width'],tuple(j['commits']))
            planner=lambda _: (job,)
        else:
            policy=req['policy']
            if policy in ('exact','row_spans'):planner=lambda r:plan(r,policy)
            elif policy=='grouped':planner=grouped_rows
            elif policy=='per_row':planner=lambda r:per_row(r,lookup_cost(req['costs']))
            elif policy=='affine_partition':planner=lambda r:partition(r,affine_cost(req['coefficients']))
            elif policy=='lookup_partition':planner=lambda r:partition(r,lookup_cost(req['costs']))
            else:raise ValueError('unknown policy')
        out,m=execute(b.fixture,rows,planner,b,mode)
    # Anything after a complete output is a validation failure, not permission
    # to lose the completed array through the native-worker exception channel.
    try:
        if out.shape!=(720,1280,3) or not np.isfinite(out).all():raise ValueError('invalid full output')
        if kind!='reference':
            check=output_integrity(out,b.strong_reference,b.baseline,rows)
            mask=np.array([[x=='1' for x in row] for row in rows])
            tile_errors=[]
            for r,c in zip(*np.nonzero(mask)):
                area=np.s_[r*120:(r+1)*120,c*128:(c+1)*128]
                tile_errors.append(float(np.abs(out[area].astype(np.float64)-b.strong_reference[area]).mean()))
            m.update(integrity=check,max_tile_mae=max(tile_errors),per_tile_mae=tile_errors)
            if b.cell=='fp16':
                selected=np.repeat(np.repeat(mask,120,axis=0),128,axis=1)
                m['selected_full_reference_mae']=float(np.abs(out[selected].astype(np.float64)-b.reference[selected]).mean())
                if m['selected_full_reference_mae']>.001:raise ValueError('FP16 independent reference threshold')
        m['passed']=True
    except Exception as error:
        m['passed']=False;m['validation_failure']=repr(error)
    return out,m


def fit_profiles(records):
    answer={}
    for reconstruction in ('whole','selected'):
        profile={};points=[];pertiles=[]
        for rclass in ('top','middle','bottom'):
            for width in WIDTHS:
                subset=[r['metrics']['per_job'][0] for r in records if r['request']['reconstruction']==reconstruction and row_class(r['request']['job']['y'])==rclass and r['request']['job']['width']==width]
                if len(subset)!=3:raise ValueError('incomplete calibration')
                bases=[];commits=[]
                for p in subset:
                    base=p['extraction_ns']+p['inference_ns']+(p['normalization_ns'] if reconstruction=='whole' else 0)
                    commit=p['scatter_ns']+(p['normalization_ns'] if reconstruction=='selected' else 0)
                    bases.append(base/1e6);commits.append(commit/len(p['commits'])/1e6)
                profile[f'{rclass}:{width}']={'base_ms':median(bases),'commit_tile_ms':median(commits)}
                points.append((width/64,median(bases)));pertiles.extend(commits)
        x=np.asarray([[1,w] for w,y in points]);y=np.asarray([y for w,y in points])
        candidates=[]
        for active in ((0,),(1,),(0,1)):
            z=np.linalg.lstsq(x[:,active],y,rcond=None)[0]
            if np.any(z<0):continue
            beta=np.zeros(2);beta[list(active)]=z
            candidates.append((float(np.sum((x@beta-y)**2)),beta))
        _,beta=min(candidates,key=lambda pair:pair[0])
        answer[reconstruction]={'costs':profile,'coefficients':[float(beta[0]),float(beta[1]),median(pertiles)],
            'training':'same per-job extraction/inference/normalization/scatter observations',
            'shared_fallback':'excluded from additive jobs; charged once in actual service',
            'approximation':'row class and width only; origin-dependent extraction and request interference omitted'}
    return answer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',choices=('width24','fp16'),required=True)
    ap.add_argument('--source-manifest-sha',required=True);ap.add_argument('--run-id',required=True)
    ap.add_argument('--guard-only',action='store_true')
    a=ap.parse_args();here=Path(__file__).resolve().parent
    if not a.run_id.startswith('E001_') or not a.run_id.replace('_','').isalnum():raise ValueError('invalid fresh identity')
    raw=(here/'source_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.source_manifest_sha:raise ValueError('source manifest drift')
    for name,pin in json.loads(raw)['files'].items():
        if hashlib.sha256((here/name).read_bytes()).hexdigest()!=pin:raise ValueError('source changed:'+name)
    _,pins,_=verify(BASE,PIN,a.cell,'engineering',0)
    expected=invariant_projection(pins['qualified_control_invariants'])
    if a.guard_only:
        current=ownership.attach(collect(),*ownership.proof(),{os.getpid()})
        print(json.dumps({'expected_invariants':expected,'current':current,'invariants_match':digest(current['invariants'])==digest(expected)},sort_keys=True))
        return
    outdir=here.parent/a.run_id;outdir.mkdir(exist_ok=False)
    worker=None;qualify.OWNED_PIDS={os.getpid()}
    with (outdir/'journal.jsonl').open('x',buffering=1) as log:
        def emit(event):log.write(json.dumps(dict(event,unix_time=time.time()),sort_keys=True,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
        def controls(stage):
            c=ownership.attach(collect(),*ownership.proof(),qualify.OWNED_PIDS);emit({'state':'control','stage':stage,'snapshot':c})
            if c['temperature_c']>=80 or c['npu_ownership_complete'] is not True or digest(c['invariants'])!=digest(expected):raise ValueError('unsafe/changed control')
            if worker is not None and not any(row['pid']==worker.process.pid for row in c['npu_ownership_proof']['owners']):raise ValueError('owned worker has no authenticated RKNPU descriptor')
            return c
        def action(req):
            controls('before_'+req['kind'])
            output,m=worker.sample(sample,req)
            arrsha=hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest()
            path=outdir/('raw_'+arrsha+'.npz')
            if not path.exists():np.savez_compressed(path,output=output)
            record={'state':'sample','request':req,'metrics':m,'array_sha256':arrsha,'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            emit(record)
            if m.get('passed') is not True:raise ValueError('numerical/runtime validation failed; raw array retained')
            controls('after_'+req['kind']);return record
        try:
            emit({'state':'running','purpose':'development_strong_baseline_screen','cell':a.cell,'source_manifest_sha':a.source_manifest_sha,'accepted_bundle_sha':PIN})
            deadline=time.monotonic()+300
            while controls('start')['temperature_c']>60:
                if time.monotonic()>deadline:raise ValueError('bounded initial cooldown exhausted')
                time.sleep(5)
            t=time.perf_counter();worker=BackendWorker(Factory(a.cell));qualify.OWNED_PIDS.add(worker.process.pid)
            emit({'state':'loaded','initialization_warmup_s':time.perf_counter()-t});controls('loaded')
            action({'kind':'reference'})
            # Physically qualify wide singleton commits at both horizontal and
            # vertical edges before interpreting any timing. Not held out.
            for y,width,right in itertools.product((0,300),WIDTHS,(False,True)):
                x=640-width if right else 0;col=9 if right else 0
                job={'x':x,'y':y,'width':width,'commits':[y//60*10+col]}
                rows=mask_from_job(Job(x,y,width,tuple(job['commits'])))
                pair=[]
                for reconstruction in ('whole','selected'):
                    pair.append(action({'kind':'qualification','rows':rows,'job':job,'reconstruction':reconstruction}))
                if pair[0]['array_sha256']!=pair[1]['array_sha256']:raise ValueError('reconstruction changed output bytes')
            profiles=[]
            options=list(itertools.product((0,120,300),WIDTHS,('whole','selected')))
            for repeat in range(3):
                shifted=options[repeat*11:]+options[:repeat*11]
                for y,width,reconstruction in shifted:
                    job={'x':0,'y':y,'width':width,'commits':[y//60*10+i for i in range(width//64)]}
                    rows=mask_from_job(Job(0,y,width,tuple(job['commits'])))
                    profiles.append(action({'kind':'profile','repeat':repeat,'rows':rows,'job':job,'reconstruction':reconstruction}))
            fitted=fit_profiles(profiles);emit({'state':'profile_frozen','profile':fitted,'records':len(profiles)})
            masks=[m for m in development_masks() if m['selected_tiles'] in (12,36) and m['mask_id'].endswith('_0')]
            policies=list(itertools.product(('exact','row_spans','grouped','per_row','affine_partition','lookup_partition'),('whole','selected')))
            n=0
            for repeat in range(3):
                for mi,mask in enumerate(masks):
                    offset=(repeat+mi)%len(policies)
                    for policy,reconstruction in policies[offset:]+policies[:offset]:
                        action(dict(mask,kind='development',repeat=repeat,policy=policy,reconstruction=reconstruction,**fitted[reconstruction]));n+=1
            worker.close();worker=None;qualify.OWNED_PIDS={os.getpid()};controls('closed')
            emit({'state':'complete','observations':n,'profiles':len(profiles),'confirmation':False})
        except BaseException as error:emit({'state':'failed','error':repr(error)});raise
        finally:
            if worker is not None:worker.close()


if __name__=='__main__':main()
