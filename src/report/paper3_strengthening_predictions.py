"""Descriptive E002 frozen-model diagnostics; no fitting or outcome-based gates.

Run only after the separate complete saved-record and numerical audits. Models
predict measured job components or normalization, not rendering or full service.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from src.report.paper3_strengthening_confirmation import ACCEPTED_SOURCE_SHA
from src.report.paper3_strengthening_development import WIDTHS

COMPONENTS = ('extraction_ns', 'inference_ns', 'normalization_ns', 'scatter_ns')
FIXED = tuple(f'{p}:{m}' for p in ('exact', 'row_spans', 'grouped', 'per_row',
                                  'affine_partition', 'lookup_partition')
              for m in ('whole', 'selected'))
FULL_REFERENCE_SHA = 'e466ed3bcaa765f95e71d4f7be8fd0ffcdffdf94d18819f47ef0f73582eb2674'


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid nonnegative numeric input')
    return float(value)


def total(values):
    result = 0.
    for value in values:
        result += number(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite accumulated input')
    return result


def predict(jobs, mode, profile, coefficients):
    if mode not in ('whole', 'selected'):
        raise ValueError('invalid reconstruction')
    costs = {row: [] for row in range(0, 360, 60)}
    call_ms = number(coefficients['call_ms'])
    element_ms = number(coefficients['million_elements_ms'])
    calls = elements = selected_elements = unselected_core_elements = halo_elements = 0
    for job in jobs:
        if (type(job['y']) is not int or job['y'] not in costs or type(job['width']) is not int or
                job['width'] not in WIDTHS or type(job['output_elements']) is not int or
                job['output_elements'] <= 0 or not job['commits']):
            raise ValueError('invalid frozen job extent')
        kind = 'top' if job['y'] == 0 else 'bottom' if job['y'] == 300 else 'middle'
        entry = profile['costs'][f'{kind}:{job["width"]}']
        count = len(job['commits'])
        costs[job['y']].append(total((number(entry['base_ms']), number(entry['commit_tile_ms']) * count)))
        calls += 1 if mode == 'whole' else count
        elements += job['output_elements'] if mode == 'whole' else count * 46080
        selected_elements += count * 46080
        core_elements = job['width'] * 60 * 12
        if core_elements < count * 46080 or job['output_elements'] < core_elements:
            raise ValueError('invalid selected/core/halo element accounting')
        unselected_core_elements += core_elements-count*46080
        halo_elements += job['output_elements']-core_elements
    return {'predicted_job_components_ms': total(total(row) for row in costs.values()),
            'predicted_normalization_ms': total((call_ms * calls, element_ms * elements / 1e6)),
            'normalization_calls': calls, 'materialized_elements': elements,
            'selected_elements': selected_elements,
            'unselected_core_elements': unselected_core_elements,
            'halo_elements': halo_elements}


def errors(rows, prediction, observed):
    differences = [r[prediction] - r[observed] for r in rows]
    return {'n': len(rows), 'mae_ms': mean(abs(x) for x in differences),
            'bias_ms': mean(differences), 'rmse_ms': mean(x*x for x in differences)**.5}


def amortization(setup_ms, saving_ms):
    if setup_ms < 0 or not math.isfinite(setup_ms) or not math.isfinite(saving_ms):
        raise ValueError('invalid setup/saving')
    return math.ceil(setup_ms / saving_ms) if saving_ms > 0 else None


def analyze(samples, profiles, coefficients, preparation_ms, calibration_service_ms):
    if len(preparation_ms) != 2:
        raise ValueError('exactly two separately initialized session preparation costs required')
    preparation_ms = [number(value) for value in preparation_ms]
    calibration_service_ms = number(calibration_service_ms)
    observations = []
    lookup = {}
    for sample in samples:
        req, met = sample['request'], sample['metrics']
        if req['kind'] != 'evaluation':
            continue
        key = req['session'], req['repeat'], req['mask_id'], req['configuration']
        if key in lookup:
            raise ValueError('duplicate evaluation observation')
        mode = met['chosen_reconstruction']
        chosen = met['chosen_policy']+':'+mode
        if chosen not in FIXED or (req['configuration'] != 'frozen_choice' and req['configuration'] != chosen):
            raise ValueError('recorded fixed/chosen configuration mismatch')
        service_ns, decision_ns = number(met['service_ns']), number(met['decision_ns'])
        components_ns = total(met[k] for k in COMPONENTS)
        if service_ns <= 0 or components_ns > service_ns or decision_ns > service_ns:
            raise ValueError('invalid complete-service component boundary')
        for name in COMPONENTS:
            if total(job[name] for job in met['per_job']) != number(met[name]):
                raise ValueError('per-job components disagree with aggregate')
        for job in met['per_job']:
            if total(job[k] for k in COMPONENTS) > number(job['job_ns']):
                raise ValueError('job components exceed elapsed job time')
        pred = predict(met['per_job'], mode, profiles[mode], coefficients)
        row = dict(zip(('session', 'repeat', 'mask_id', 'configuration'), key)) | pred | {
            'chosen_configuration': chosen,
            'observed_job_components_ms': components_ns/1e6,
            'observed_normalization_ms': met['normalization_ns']/1e6,
            'service_ms': service_ns/1e6,
            'decision_ms': decision_ns/1e6,
        }
        lookup[key] = row
        observations.append(row)
    masks = sorted({r['mask_id'] for r in observations})
    expected = {(s, r, m, c) for s in (0, 1) for r in (0, 1) for m in masks
                for c in (*FIXED, 'frozen_choice')}
    if len(masks) != 16 or set(lookup) != expected:
        raise ValueError('incomplete two-session confirmation matrix')
    def pooled(mask, config):
        return mean(lookup[s, r, mask, config]['service_ms'] for s in (0,1) for r in (0,1))
    fixed_means = {c: mean(pooled(m,c) for m in masks) for c in FIXED}
    best = min(FIXED, key=fixed_means.get)
    choice_rows = [r for r in observations if r['configuration'] == 'frozen_choice']
    choice_mean = mean(r['service_ms'] for r in choice_rows)
    wrong_choices = []
    for mask in masks:
        selected = {lookup[s,r,mask,'frozen_choice']['chosen_configuration'] for s in (0,1) for r in (0,1)}
        if len(selected) != 1:
            raise ValueError('frozen choice changed across repetitions')
        chosen = selected.pop()
        observed_best = min(FIXED, key=lambda c: pooled(mask,c))
        wrong_choices.append({'mask_id': mask, 'chosen': chosen,
            'observed_best_fixed_for_mask': observed_best,
            'measured_choice_penalty_ms': pooled(mask,chosen)-pooled(mask,observed_best),
            'measured_choice_penalty_percent': 100*(pooled(mask,chosen)/pooled(mask,observed_best)-1),
            'actual_chooser_minus_hindsight_best_ms': pooled(mask,'frozen_choice')-pooled(mask,observed_best),
            'chooser_service_ms': pooled(mask,'frozen_choice'),
            'chosen_fixed_service_ms': pooled(mask,chosen)})
    pairs = []
    for session in (0,1):
        for repeat in (0,1):
            for mask in masks:
                for policy in ('exact','row_spans','grouped'):
                    a, b = (lookup[session,repeat,mask,policy+':'+m] for m in ('whole','selected'))
                    predicted = a['predicted_normalization_ms']-b['predicted_normalization_ms']
                    observed = a['observed_normalization_ms']-b['observed_normalization_ms']
                    profiled = a['predicted_job_components_ms']-b['predicted_job_components_ms']
                    component_observed = a['observed_job_components_ms']-b['observed_job_components_ms']
                    service_observed = a['service_ms']-b['service_ms']
                    volume = a['materialized_elements']-b['materialized_elements']
                    pairs.append({'session': session, 'repeat': repeat, 'mask_id': mask,
                        'policy': policy, 'predicted_saving_ms': predicted,
                        'observed_saving_ms': observed,
                        'prediction_error_ms': predicted-observed,
                        'profile_predicted_component_saving_ms': profiled,
                        'observed_component_saving_ms': component_observed,
                        'observed_service_saving_ms': service_observed,
                        'discarded_materialized_elements': volume,
                        'volume_only_sign_agrees': (volume > 0)-(volume < 0) == (observed > 0)-(observed < 0),
                        'profile_component_sign_agrees': (profiled > 0)-(profiled < 0) == (component_observed > 0)-(component_observed < 0),
                        'profile_service_sign_agrees': (profiled > 0)-(profiled < 0) == (service_observed > 0)-(service_observed < 0),
                        'sign_agrees': (predicted > 0)-(predicted < 0) == (observed > 0)-(observed < 0)})
    normalization_choices = []
    for mask in masks:
        for policy in ('exact','row_spans','grouped'):
            a,b=(lookup[0,0,mask,policy+':'+m] for m in ('whole','selected'))
            observed = {m: pooled(mask,policy+':'+m) for m in ('whole','selected')}
            oracle = min(observed.values())
            candidates = {'call_element': 'selected' if a['predicted_normalization_ms'] > b['predicted_normalization_ms'] else 'whole',
                          'profiled_job_components': 'selected' if a['predicted_job_components_ms'] > b['predicted_job_components_ms'] else 'whole',
                          'volume_only': 'selected' if a['materialized_elements'] > b['materialized_elements'] else 'whole',
                          'always_whole': 'whole', 'always_selected': 'selected'}
            normalization_choices.append({'mask_id':mask,'policy':policy,
                'observed_whole_service_ms':observed['whole'],'observed_selected_service_ms':observed['selected'],
                'choices': {name:{'mode':mode,'pooled_measured_wrong_choice_ms':observed[mode]-oracle}
                            for name,mode in candidates.items()}})
    saving = fixed_means[best]-choice_mean
    setup = mean(preparation_ms)
    return {'fixed_means_ms': fixed_means, 'best_fixed': best,
        'chooser_mean_ms': choice_mean, 'chooser_decision_mean_ms': mean(r['decision_ms'] for r in choice_rows),
        'job_cost_prediction': errors(observations,'predicted_job_components_ms','observed_job_components_ms'),
        'normalization_prediction': errors(observations,'predicted_normalization_ms','observed_normalization_ms'),
        'normalization_pair_prediction_mae_ms': mean(abs(p['prediction_error_ms']) for p in pairs),
        'normalization_pair_sign_agreement': mean(p['sign_agrees'] for p in pairs),
        'always_selected_sign_agreement': mean(p['observed_saving_ms'] > 0 for p in pairs),
        'always_whole_sign_agreement': mean(p['observed_saving_ms'] < 0 for p in pairs),
        'volume_only_sign_agreement': mean(p['volume_only_sign_agrees'] for p in pairs),
        'profile_component_sign_agreement': mean(p['profile_component_sign_agrees'] for p in pairs),
        'profile_service_sign_agreement': mean(p['profile_service_sign_agrees'] for p in pairs),
        'profile_pair_component_prediction_mae_ms': mean(abs(p['profile_predicted_component_saving_ms']-p['observed_component_saving_ms']) for p in pairs),
        'normalization_choices': normalization_choices,
        'wrong_choices': wrong_choices, 'normalization_pairs': pairs,
        'preparation_ms': preparation_ms, 'calibration_measured_service_ms': calibration_service_ms,
        'conditional_amortization': {
            'saving_vs_best_fixed_ms': saving,
            'total_preparation_requests': amortization(setup,saving),
            'preparation_plus_calibration_service_requests': amortization(setup+calibration_service_ms,saving),
            'interpretation': 'Conditional on repeating this workload after one initialization; uses mean of two session preparations. Total charged-preparation illustration, not incremental setup versus calibrated fixed planners. Calibration/qualification service excludes control/archival/warm-up wall time, export/compilation and runtime/model initialization; not total deployment cost.'},
        'limits': ['No coefficient refitting or additional acceptance gate.',
                   'Prediction boundary excludes common fallback, planning, decision and loop gaps.',
                   'Measured best fixed per mask is a hindsight oracle from separate noisy executions, not a counterfactual.',
                   'Measured choice penalty compares the chosen fixed configuration with the hindsight best; separately timed chooser cost can differ through decision work and ordinary timing variation.',
                   'Additional normalization-choice rules are offline descriptive choices over fixed measurements, not separately timed deployed policies; their decision overhead is not measured.',
                   'Preparation serialized bytes are not resident memory.'],
        'observations': observations}


def source_binding(bundle, audit):
    """Bind secondary analysis to the accepted source and four completed audits."""
    bundle = Path(bundle).resolve()
    manifest_raw = (bundle/'source_manifest.json').read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != ACCEPTED_SOURCE_SHA:
        raise ValueError('secondary analysis source is not accepted E002 v3')
    manifest = json.loads(manifest_raw)
    for name, pin in manifest['files'].items():
        path = bundle/name
        if path.is_symlink() or not path.resolve().is_relative_to(bundle) or hashlib.sha256(path.read_bytes()).hexdigest() != pin:
            raise ValueError('frozen source file changed or escaped bundle')
    frozen_raw = (bundle/'frozen.json').read_bytes()
    frozen = json.loads(frozen_raw)
    frozen_sha = hashlib.sha256(frozen_raw).hexdigest()
    method_raw = (bundle/'method_freeze.json').read_bytes()
    if hashlib.sha256(method_raw).hexdigest() != frozen['method_freeze_sha256']:
        raise ValueError('method binding changed')
    method = json.loads(method_raw)
    runs = audit.get('runs', [])
    if audit.get('status') != 'complete_saved_record_audit' or len(runs) != 4:
        raise ValueError('four completed authentication records required')
    authenticated = {(r['cell'], r['session']): r for r in runs}
    if set(authenticated) != {(c, s) for c in ('width24', 'fp16') for s in (0, 1)}:
        raise ValueError('duplicate or incomplete authenticated cells')
    if len({r['journal_sha256'] for r in runs}) != 4:
        raise ValueError('each authenticated cell requires a distinct journal')
    if any(r['source_manifest_sha256'] != ACCEPTED_SOURCE_SHA or r['frozen_sha256'] != frozen_sha for r in runs):
        raise ValueError('completed audits use another source or freeze')
    return frozen, frozen_sha, method, authenticated


def bind_numerical_reports(reports, authenticated):
    """Verify existing replay certificates; this does not rerun tensor checks."""
    expected = {r['journal_sha256']: key for key, r in authenticated.items()}
    if len(reports) != 4 or len(expected) != 4:
        raise ValueError('four numerical reports required')
    found = {}
    for report in reports:
        journal = report.get('journal_sha256')
        if journal not in expected or journal in found:
            raise ValueError('unexpected or duplicate numerical journal binding')
        cell, session = expected[journal]
        if (report.get('status') != 'retained_pixel_replay_passed' or report.get('confirmation') is not True or
                report.get('sample_count') != 465 or report.get('cell') != cell or
                report.get('session') != session or report.get('source_manifest_sha256') != ACCEPTED_SOURCE_SHA or
                report.get('fixed_job_reconstruction_pairs') != 96 or
                report.get('fixed_job_reconstruction_pairs_byte_identical') is not True):
            raise ValueError('incomplete or unsuccessful numerical confirmation')
        if (report.get('independent_fp16_reference_used') is not (cell == 'fp16') or
                report.get('full_reference_file_sha256') != (FULL_REFERENCE_SHA if cell == 'fp16' else None)):
            raise ValueError('independent numerical reference binding differs')
        found[journal] = (cell, session)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--numerical-audit', type=Path, action='append', required=True)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--development-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit_raw = args.audit.read_bytes()
    audit = json.loads(audit_raw)
    frozen, frozen_sha, method, authenticated = source_binding(args.bundle, audit)
    numerical_raw = [path.read_bytes() for path in args.numerical_audit]
    numerical_reports = [json.loads(raw) for raw in numerical_raw]
    numerical_bindings = bind_numerical_reports(numerical_reports, authenticated)
    groups = {cell: [] for cell in ('width24','fp16')}
    prep = {cell: [] for cell in groups}
    found = set()
    for directory in args.run:
        raw = (directory/'journal.jsonl').read_bytes()
        events = [json.loads(line) for line in raw.splitlines()]
        first = events[0];key=first['cell'],first['session']
        if key in found or key not in authenticated:
            raise ValueError('unexpected/duplicate audited run')
        found.add(key)
        expected = authenticated[key]
        if hashlib.sha256(raw).hexdigest() != expected['journal_sha256'] or frozen_sha != expected['frozen_sha256']:
            raise ValueError('journal/frozen data differ from completed audit')
        groups[key[0]].extend(e for e in events if e['state']=='sample')
        preparation = next(e for e in events if e['state']=='preparation')
        prep[key[0]].append(preparation['preparation_ns']/1e6)
    if len(found) != 4:
        raise ValueError('four authenticated cells required')
    results = {}
    for cell in groups:
        development = args.development_root/f'E001_{cell}_v5'/'journal.jsonl'
        raw = development.read_bytes()
        if hashlib.sha256(raw).hexdigest() != method['calibration_bindings'][cell]['journal_sha256']:
            raise ValueError('original calibration journal changed')
        events = [json.loads(line) for line in raw.splitlines()]
        calibration = [e for e in events if e['state']=='sample' and e['request']['kind'] in ('qualification','profile')]
        if len(calibration) != 156:
            raise ValueError('incomplete original calibration')
        calibration_service = total(e['metrics']['service_ns'] for e in calibration)/1e6
        results[cell] = analyze(groups[cell], frozen['profiles'][cell],
                                frozen['materialization_models'][cell], prep[cell], calibration_service)
    with args.output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps({'graphs': results, 'audit_sha256': hashlib.sha256(audit_raw).hexdigest(),
                                'source_manifest_sha256': ACCEPTED_SOURCE_SHA,
                                'frozen_sha256': frozen_sha,
                                'numerical_audits': [{'sha256': hashlib.sha256(raw).hexdigest(),
                                    'journal_sha256': report['journal_sha256'],
                                    'cell': numerical_bindings[report['journal_sha256']][0],
                                    'session': numerical_bindings[report['journal_sha256']][1]}
                                    for raw, report in zip(numerical_raw, numerical_reports, strict=True)]},
                               indent=2,sort_keys=True,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
