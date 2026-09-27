"""Once-only method reservation, named exclusion catalog, masks and source overlay.

Does not contact a board, compile models, fit new outcomes or inspect future
evaluation. Exclusive output directory prevents replacing a previous freeze.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.experiments.paper3_strengthening_v2.cached import compile_choice,CANDIDATES
from src.experiments.paper3_strengthening_v2.confirmation import matched_masks,schedule,MASK_SEED,ORDER_SEED

FILES={'run_e001.py':'results/paper3_strengthening_v2/E001_source_v5/run_e001.py',
       'run_e002.py':'scripts/paper3_strengthening_v2/run_e002.py',
       'strength/__init__.py':'src/experiments/paper3_strengthening_v2/__init__.py',
       **{'strength/'+name+'.py':'src/experiments/paper3_strengthening_v2/'+name+'.py'
          for name in ('core','service','ownership','cached','confirmation','evaluation')},
       'protocol.md':'docs/paper3_strengthening_v2/E002_DRAFT.md',
       'build_e002.py':'scripts/paper3_strengthening_v2/build_e002.py',
       'headless_e002_template.py':'scripts/paper3_strengthening_v2/headless_e002_template.py'}
CATALOGS=('results/paper3_transfer_v1/method_freeze.json','results/paper3_transfer_v1/plan.json',
          'results/paper3_dispatch_successor_v1/development_inventory.json',
          'results/paper3_dispatch_successor_v1/confirmation_plan.json',
          'results/tables/paper3_npu_fragmentation_summary_v1.json',
          'configs/paper3_rk3576_fragmentation_pairs_v1.yaml',
          'output/paper3_transfer_v1/bundle_check_v2/engineering_plan.json',
          'output/paper3_packing_v1/E002_width24/journal.jsonl',
          'output/paper3_packing_v1/E002_fp16/journal.jsonl',
          'results/paper3_strengthening_v2/E001_width24_v5/journal.jsonl',
          'results/paper3_strengthening_v2/E001_fp16_v5/journal.jsonl')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def geometry(value):
    """Only explicit six-row bit strings and mask_hex fields, never timings."""
    found=set()
    if isinstance(value,(tuple,list)):
        if len(value)==6 and all(isinstance(row,str) and len(row)==10 and set(row)<={'0','1'} for row in value):
            found.add(tuple(value))
        else:
            for item in value:found.update(geometry(item))
    elif isinstance(value,dict):
        for key,item in value.items():
            if key=='mask_hex':
                number=int(item,16)
                if number<0 or number>=2**60:raise ValueError('invalid historical mask')
                found.add(tuple(''.join('1' if number & (1<<(r*10+c)) else '0' for c in range(10)) for r in range(6)))
            else:found.update(geometry(item))
    return found


def exclusions():
    import yaml
    union=set();sources=[]
    for rel in CATALOGS:
        path=ROOT/rel;raw=path.read_bytes()
        if path.suffix=='.jsonl':value=[json.loads(line) for line in raw.splitlines() if line.strip()]
        elif path.suffix=='.yaml':value=yaml.safe_load(raw)
        else:value=json.loads(raw)
        rows=geometry(value)
        if not rows:raise ValueError('named catalog yielded no explicit geometry:'+rel)
        union.update(rows);sources.append({'path':rel,'sha256':sha(raw),'unique_geometries':len(rows)})
    rel='output/paper3_packing_v1/E001_source/qualify_rk3566.py';raw=(ROOT/rel).read_bytes()
    if b"ROWS=('1111111111',)*6" not in raw:raise ValueError('packing E001 full mask source changed')
    union.add(('1111111111',)*6)
    sources.append({'path':rel,'sha256':sha(raw),'unique_geometries':1,'extraction':'explicit ROWS full-mask literal'})
    return {'claim':'Non-overlap with these named catalogs only, not a universal claim about all historical requests.',
            'sources':sources,'unique_geometries':len(union),'rows':sorted(union)}


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--identity',required=True)
    parser.add_argument('--inherit-reservation',type=Path)
    args=parser.parse_args()
    if not args.identity.startswith('E002_source_') or not args.identity.replace('_','').isalnum():raise ValueError('bad identity')
    output=ROOT/'results/paper3_strengthening_v2'/args.identity;output.mkdir(exist_ok=False)
    files={}
    for rel,source in FILES.items():
        target=output/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/source,target)
        files[rel]=sha(target.read_bytes())
    excluded=exclusions();write(output/'exclusion_catalog.json',excluded)
    profiles={};models={};bindings={};compiled={}
    for cell in ('width24','fp16'):
        path=ROOT/'results/paper3_strengthening_v2'/('E001_'+cell+'_v5')/'journal.jsonl'
        raw=path.read_bytes();events=[json.loads(line) for line in raw.splitlines() if line.strip()]
        if events[-1]['state']!='complete':raise ValueError('unfinished development calibration')
        selected=[event['profile'] for event in events if event['state']=='profile_frozen']
        if len(selected)!=1:raise ValueError('ambiguous calibration')
        profiles[cell]=selected[0]
        modelpath=path.parent.parent/('E001_'+cell+'_v5_materialization_model.json')
        modelraw=modelpath.read_bytes();models[cell]=json.loads(modelraw)['coefficients']
        bindings[cell]={'journal_path':str(path.relative_to(ROOT)).replace('\\','/'),'journal_sha256':sha(raw),
                        'model_path':str(modelpath.relative_to(ROOT)).replace('\\','/'),'model_sha256':sha(modelraw)}
        choice=compile_choice(profiles[cell])
        compiled[cell]={'score_sha256':choice.score_sha256,'serialized_score_bytes':choice.serialized_score_bytes,
            'tables':{p+':'+m:{key:getattr(table,key) for key in ('table_sha256','cost_sha256','serialized_table_bytes')}
                      for (p,m),table in choice.planners.items()}}
    # This record is durable BEFORE reserving the first evaluation geometry.
    method={'identity':args.identity,'unix_time':time.time(),'mask_seed':MASK_SEED,'order_seed':ORDER_SEED,
            'files':files,'exclusion_catalog_sha256':sha((output/'exclusion_catalog.json').read_bytes()),
            'calibration_bindings':bindings,'profiles':profiles,'materialization_models':models,'compiled':compiled,
            'candidate_order':[p+':'+m for p,m in CANDIDATES],
            'chooser_objective':'Frozen calibrated additive job costs; common fallback cancels; no obsolete online DP penalty. Actual decision/planning is measured.',
            'practical_gates':{'chooser_mean_gain_fraction':.05,'chooser_positive_each_session':True,
                'maximum_pooled_mask_regression_fraction':.10,'primary_geometry_policy':'row_spans',
                'geometry_minimum_ms':2.,'geometry_fraction_whole_compact':.05,'geometry_minimum_families_per_graph':2},
            'bootstrap':{'draws':5000,'seed':20260925,'unit':'matched pair within family; preserve both members/sessions/repeats',
                'best_fixed_reselected_each_draw':True,'percentiles':[2.5,97.5],
                'interpretation':'Descriptive sensitivity interval, not validated population 95% coverage; only two pairs per family.'}}
    previous=None
    if args.inherit_reservation is not None:
        predecessor=args.inherit_reservation.resolve();oldraw=(predecessor/'frozen.json').read_bytes()
        previous=json.loads(oldraw)
        if previous['profiles']!=profiles or previous['materialization_models']!=models:
            raise ValueError('inherited reservation cannot refit profiles/models')
        if json.loads((predecessor/'exclusion_catalog.json').read_bytes())!=json.loads(json.dumps(excluded)):
            raise ValueError('inherited reservation exclusion catalog changed')
        method['reservation_inherited_from']={'path':str(predecessor),'frozen_sha256':sha(oldraw),
            'reason':'Zero-sample E002 v1 preparation failure; explicit float accumulation portability correction only.'}
    write(output/'method_freeze.json',method)
    if previous is None:
        masks=matched_masks(excluded['rows']);requests=schedule(masks)
    else:
        masks=previous['masks'];requests=previous['schedule']
        if json.loads(json.dumps(matched_masks(excluded['rows'])))!=masks or schedule(masks)!=requests:
            raise ValueError('inherited masks/schedule differ from frozen generation')
    frozen={'method_freeze_sha256':sha((output/'method_freeze.json').read_bytes()),'masks':masks,'schedule':requests,
            'profiles':profiles,'compiled':compiled,'materialization_models':models,
            'evaluation_requests':1664,'qualification_requests':192,'reference_requests':4,
            'reserved_unix_time':previous['reserved_unix_time'] if previous else time.time()}
    write(output/'frozen.json',frozen)
    for name in ('exclusion_catalog.json','method_freeze.json','frozen.json'):files[name]=sha((output/name).read_bytes())
    write(output/'source_manifest.json',{'purpose':'paper3_strengthening_v2_E002_confirmation','files':files})
    archive=output.parent/(args.identity+'.tar')
    with tarfile.open(archive,'x') as stream:
        for path in sorted(output.rglob('*')):
            if path.is_file():stream.add(path,arcname=path.relative_to(output).as_posix())
    launcher=output.parent/(args.identity+'_launcher.py')
    source_sha=sha((output/'source_manifest.json').read_bytes())
    template=(output/'headless_e002_template.py').read_text()
    with launcher.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(template.replace('SOURCE_ID_TOKEN',args.identity).replace('SOURCE_SHA_TOKEN',source_sha))
    print(json.dumps({'directory':str(output),'source_manifest_sha256':sha((output/'source_manifest.json').read_bytes()),
                     'archive_sha256':sha(archive.read_bytes()),'excluded_unique_masks':excluded['unique_geometries'],
                     'reserved_masks':len(masks),'evaluation_requests':len(requests),'launcher_sha256':sha(launcher.read_bytes())}))


if __name__=='__main__':main()
