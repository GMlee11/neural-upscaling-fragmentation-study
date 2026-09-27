"""Compact transfer replay; archived raw-audit attestations are not rerun here."""
import argparse
import json
import math
from pathlib import Path

from src.report.paper3_dispatch_publication_v1 import features
from src.experiments.paper3_transfer_v1.core import analyze, COEFFICIENTS

ROOT = Path(__file__).resolve().parents[2]
INPUT = 'results/manifests/paper3_transfer_public_v1.json'


def summarize(payload):
    if payload['artifact'] != 'paper3_transfer_public_v1' or payload['donor']['coefficients'] != list(COEFFICIENTS):
        raise ValueError('wrong artifact or donor')
    if payload['donor']['additional_target_calibration_requests'] != 0:
        raise ValueError('target fitting forbidden')
    results = []
    if [(c['device'], c['graph']) for c in payload['cells']] != [(d,g) for d in ('rk3576','rk3566') for g in ('width24','fp16')]:
        raise ValueError('incomplete cell matrix')
    for cell in payload['cells']:
        records = []
        if len(cell['sessions']) != 2:
            raise ValueError('two sessions required')
        for index, session in enumerate(cell['sessions']):
            rows = session['records']
            if session['session'] != index or len(rows) != 640 or not session['raw_files_verified']:
                raise ValueError('incomplete audited session')
            if session['status'] != 'raw_replay_pass':
                raise ValueError('raw replay not accepted')
            expected = payload['requests'][index*640:(index+1)*640]
            for r, request in zip(rows, expected):
                if any(r[k] != request[k] for k in ('block','mask_id','order','policy','rows')):
                    raise ValueError('schedule changed')
                if not r['output_verified'] or r['status'] != 'complete':
                    raise ValueError('unverified output')
                if not all(math.isfinite(r[k]) and r[k] >= 0 for k in ('service_ms','policy_ms')) or r['service_ms'] < r['policy_ms']:
                    raise ValueError('invalid timing')
                exact, span = (features(r['rows'], mode, cell['tensor_contract']) for mode in ('exact','row_spans'))
                policy = r['policy']
                if policy == 'always_exact': mode = 'exact'
                elif policy == 'always_span': mode = 'row_spans'
                elif policy == 'geometry':
                    count = sum(row.count('1') for row in r['rows'])
                    mode = 'row_spans' if span[1] <= count/2 and round(span[2]*1e6/3840) <= count*1.25 else 'exact'
                else:
                    mode = 'row_spans' if sum(c*x for c,x in zip(COEFFICIENTS,span)) < sum(c*x for c,x in zip(COEFFICIENTS,exact)) else 'exact'
                if mode != r['chosen_mode']: raise ValueError('decision changed')
            records.extend(rows)
        full = analyze(records, payload['requests'], payload['donor']['calibration_service_ms'])
        for key in full:
            if full[key] != cell['audited_summary'][key]:
                raise ValueError('archived raw-audit summary differs: '+key)
        fixed = full['best_fixed_posthoc']
        fixed_mean = full['policies'][fixed]['mean_service_ms']
        candidate = full['policies']['calibrated']
        sessions = [analyze(s['records'], payload['requests'][i*640:(i+1)*640], payload['donor']['calibration_service_ms']) for i,s in enumerate(cell['sessions'])]
        result = {k:v for k,v in full.items() if k not in ('observed_choice_penalties','policies')}
        result.update(device=cell['device'], graph=cell['graph'], requests=1280,
            reduction_percent=100*(fixed_mean-candidate['mean_service_ms'])/fixed_mean,
            policies={p:{k:v for k,v in row.items() if k!='paired_deltas_ms'} for p,row in full['policies'].items()},
            session_reductions_percent=[100*(s['policies'][s['best_fixed_posthoc']]['mean_service_ms']-s['policies']['calibrated']['mean_service_ms'])/s['policies'][s['best_fixed_posthoc']]['mean_service_ms'] for s in sessions])
        results.append(result)
    return {'status':'COMPACT_TRANSFER_REPLAY_PASS','samples':5120,'cells':results,
            'scope':'Zero target fitting on previously studied graph/device cells; fresh requests at method freeze, later cells reuse opened geometries. Compact replay does not re-audit tensors.'}


def latex(report):
    lines = [r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Device & Graph & Exact & Row-span & Geometry & Fixed rule & Reduction \\', r'\midrule']
    for c in report['cells']:
        name = 'Width-24 INT8' if c['graph']=='width24' else 'Seven-conv. FP16'
        means = [c['policies'][p]['mean_service_ms'] for p in ('always_exact','always_span','geometry','calibrated')]
        lines.append(c['device'].upper()+' & '+name+' & '+' & '.join(f'{x:.2f}' for x in means)+f" & {c['reduction_percent']:.2f}\\% "+r'\\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n'


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    value=summarize(json.loads((args.root/INPUT).read_bytes()))
    targets={'results/tables/paper3_transfer_publication_v1.json':json.dumps(value,indent=2,sort_keys=True)+'\n',
             'docs/paper/generated/paper3_transfer_publication_v1.tex':latex(value)}
    for name,text in targets.items():
        p=args.root/name
        if args.write:
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8',newline='\n')
        elif p.read_text(encoding='utf-8')!=text:
            raise ValueError('generated output differs: '+name)
    print(value['status'],value['samples'])
