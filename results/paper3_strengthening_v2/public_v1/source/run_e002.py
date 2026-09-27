"""Frozen E002 matched-geometry confirmation; never refits measured outcomes."""
import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np

# Frozen E001 module supplies only accepted backend, controls and qualification.
from run_e001 import BASE,PIN,verify,invariant_projection,collect,digest,qualify,ownership
from run_e001 import BackendWorker,Job,WIDTHS,mask_from_job,sample as qualify_sample
from strength.cached import compile_choice
from strength.evaluation import evaluate


@dataclass
class Factory:
    cell:str
    def __call__(self):return qualify.NumericalBackend(self.cell)


def prepare(request,backend):
    backend.choice=compile_choice(request['profile'])
    choice=backend.choice
    result={'preparation_ns':choice.preparation_ns,'score_sha256':choice.score_sha256,
            'serialized_score_bytes':choice.serialized_score_bytes,
            'tables':{p+':'+m:{key:getattr(table,key) for key in
                ('table_sha256','cost_sha256','serialized_table_bytes','preparation_ns')}
                for (p,m),table in choice.planners.items()}}
    wanted=request['expected']
    if result['score_sha256']!=wanted['score_sha256']:
        raise ValueError('frozen score table drift')
    for key,table in result['tables'].items():
        if table['table_sha256']!=wanted['tables'][key]['table_sha256'] or table['cost_sha256']!=wanted['tables'][key]['cost_sha256']:
            raise ValueError('frozen planner table drift')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',choices=('width24','fp16'),required=True)
    ap.add_argument('--session',type=int,choices=(0,1),required=True)
    ap.add_argument('--source-manifest-sha',required=True);ap.add_argument('--run-id',required=True)
    args=ap.parse_args();here=Path(__file__).resolve().parent
    if not args.run_id.startswith('E002_') or not args.run_id.replace('_','').isalnum():raise ValueError('invalid run identity')
    manifest_raw=(here/'source_manifest.json').read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest()!=args.source_manifest_sha:raise ValueError('manifest mismatch')
    for name,pin in json.loads(manifest_raw)['files'].items():
        if hashlib.sha256((here/name).read_bytes()).hexdigest()!=pin:raise ValueError('frozen source changed:'+name)
    frozen=json.loads((here/'frozen.json').read_bytes())
    requests=[req for req in frozen['schedule'] if req['session']==args.session and req['cell']==args.cell]
    if len(requests)!=416:raise ValueError('schedule cell count drift')
    _,pins,_=verify(BASE,PIN,args.cell,'engineering',0)
    expected=invariant_projection(pins['qualified_control_invariants'])
    outdir=here.parent/args.run_id;outdir.mkdir(exist_ok=False)
    worker=None;qualify.OWNED_PIDS={os.getpid()}
    with (outdir/'journal.jsonl').open('x',buffering=1) as log:
        def emit(event):
            log.write(json.dumps(dict(event,unix_time=time.time()),sort_keys=True,allow_nan=False)+'\n')
            log.flush();os.fsync(log.fileno())
        def controls(stage):
            value=ownership.attach(collect(),*ownership.proof(),qualify.OWNED_PIDS)
            emit({'state':'control','stage':stage,'snapshot':value})
            if value['temperature_c']>=80 or value['npu_ownership_complete'] is not True or digest(value['invariants'])!=digest(expected):
                raise ValueError('unsafe/changed control')
            if worker is not None and not any(owner['pid']==worker.process.pid for owner in value['npu_ownership_proof']['owners']):
                raise ValueError('owned worker lacks authenticated RKNPU descriptor')
            return value
        def action(req):
            controls('before_'+req['kind'])
            function=evaluate if req['kind']=='evaluation' else qualify_sample
            output,metrics=worker.sample(function,req)
            array_sha=hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest()
            path=outdir/('raw_'+array_sha+'.npz')
            if not path.exists():np.savez_compressed(path,output=output)
            record={'state':'sample','request':req,'metrics':metrics,'array_sha256':array_sha,
                    'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            emit(record)
            if metrics.get('passed') is not True:raise ValueError('numerical failure; completed raw array retained')
            controls('after_'+req['kind']);return record
        try:
            emit({'state':'running','purpose':'reserved_matched_geometry_confirmation','cell':args.cell,'session':args.session,
                  'source_manifest_sha':args.source_manifest_sha,'accepted_bundle_sha':PIN,
                  'frozen_sha256':hashlib.sha256((here/'frozen.json').read_bytes()).hexdigest(),
                  'method_freeze_sha256':hashlib.sha256((here/'method_freeze.json').read_bytes()).hexdigest()})
            deadline=time.monotonic()+300
            while controls('start')['temperature_c']>60:
                if time.monotonic()>deadline:raise ValueError('bounded initial cooldown exhausted')
                time.sleep(5)
            start=time.perf_counter();worker=BackendWorker(Factory(args.cell));qualify.OWNED_PIDS.add(worker.process.pid)
            emit({'state':'loaded','initialization_warmup_s':time.perf_counter()-start});controls('loaded')
            prepared=worker.sample(prepare,{'profile':frozen['profiles'][args.cell],
                                           'expected':frozen['compiled'][args.cell]})
            emit({'state':'preparation','profile_sha256':digest(frozen['profiles'][args.cell]),**prepared})
            controls('prepared')
            action({'kind':'reference','session':args.session,'cell':args.cell})
            for y,width,right in itertools.product((0,300),WIDTHS,(False,True)):
                x=640-width if right else 0;column=9 if right else 0
                job={'x':x,'y':y,'width':width,'commits':[y//60*10+column]}
                rows=mask_from_job(Job(x,y,width,tuple(job['commits'])))
                pair=[]
                for reconstruction in ('whole','selected'):
                    pair.append(action({'kind':'qualification','session':args.session,'cell':args.cell,
                        'rows':rows,'job':job,'reconstruction':reconstruction}))
                if pair[0]['array_sha256']!=pair[1]['array_sha256']:raise ValueError('qualification reconstruction byte mismatch')
            for request in requests:action(request)
            worker.close();worker=None;qualify.OWNED_PIDS={os.getpid()};controls('closed')
            emit({'state':'complete','observations':len(requests),'qualifications':48,'references':1,'confirmation':True})
        except BaseException as error:emit({'state':'failed','error':repr(error)});raise
        finally:
            if worker is not None:worker.close()


if __name__=='__main__':main()
