import copy
import itertools
import hashlib
import json
from pathlib import Path

import numpy as np

import pytest

from src.report.paper3_strengthening_confirmation import (
    ACCEPTED_BUNDLE_SHA, ACCEPTED_SOURCE_SHA, PRACTICAL_GATES, BOOTSTRAP_CONTRACT,
    analyze, authenticate_run, frozen_expectations,
)
from src.report.paper3_strengthening_development import STAGES, WIDTHS


CONFIGS = tuple(f'{p}:{m}' for p in ('exact', 'row_spans', 'grouped', 'per_row', 'affine_partition', 'lookup_partition')
                for m in ('whole', 'selected'))


def fixture():
    masks = [dict(mask_id=f'f{family}p{pair}_{member}', pair_id=f'f{family}p{pair}',
                  family=f'f{family}', member=member)
             for family, pair, member in itertools.product(range(4), range(2), ('compact', 'fragmented'))]
    records = []
    for session, repeat, mask, config in itertools.product(range(2), range(2), masks, (*CONFIGS, 'chooser')):
        ms = 90 if config == 'chooser' else 100
        if mask['member'] == 'fragmented' and config == 'row_spans:whole':
            ms = 130
        elif mask['member'] == 'fragmented' and config == 'row_spans:selected':
            ms = 110
        records.append(dict(session=session, repeat=repeat, mask_id=mask['mask_id'], configuration=config,
                            service_ns=ms*1e6, normalization_ns=20e6))
    return masks, records


def test_practical_gates_and_complete_counts():
    masks, records = fixture()
    result = analyze(records, masks, CONFIGS)
    assert result['evaluation_rows'] == 832
    assert result['best_fixed'] == 'exact:whole'
    assert result['chooser']['mean_reduction_percent'] == pytest.approx(10)
    assert result['chooser']['practical_gate_pass']
    assert result['geometry_penalty_reduction_gate_pass']
    assert result['geometry_families']['f0']['pooled_service_intervention_ms'] == 20


def test_searches_all_twelve_fixed_controls():
    masks, records = fixture()
    for row in records:
        if row['configuration'] == CONFIGS[-1]:
            row['service_ns'] = 80e6
    result = analyze(records, masks, CONFIGS)
    assert result['best_fixed'] == CONFIGS[-1]
    assert not result['chooser']['practical_gate_pass']
    assert result['chooser']['mean_reduction_percent'] == pytest.approx(-12.5)


def test_positive_pooled_mean_does_not_hide_session_regression():
    masks, records = fixture()
    for row in records:
        if row['configuration'] == 'chooser':
            row['service_ns'] = (83 if row['session'] == 0 else 105)*1e6
    result = analyze(records, masks, CONFIGS)
    assert result['chooser']['mean_reduction_percent'] == pytest.approx(6)
    assert not result['chooser']['practical_gate_pass']


def test_average_does_not_hide_large_mask_regression():
    masks, records = fixture()
    for row in records:
        if row['configuration'] == 'chooser':
            row['service_ns'] = (120 if row['mask_id'] == masks[0]['mask_id'] else 80)*1e6
    result = analyze(records, masks, CONFIGS)
    assert result['chooser']['mean_reduction_percent'] > 5
    assert result['chooser']['worst_mask_reduction_percent'] == -20
    assert not result['chooser']['practical_gate_pass']


def test_negative_intervention_is_not_a_penalty_reduction():
    masks, records = fixture()
    for row in records:
        if row['mask_id'].endswith('fragmented') and row['configuration'] == 'row_spans:selected':
            row['service_ns'] = 160e6
    result = analyze(records, masks, CONFIGS)
    assert result['geometry_magnitude_gate_pass']
    assert not result['geometry_penalty_reduction_gate_pass']
    assert result['geometry_families']['f0']['pooled_service_intervention_ms'] == -30


def test_session_reversal_fails_geometry_gate():
    masks, records = fixture()
    for row in records:
        if row['session'] == 1 and row['mask_id'].endswith('fragmented') and row['configuration'] == 'row_spans:selected':
            row['service_ns'] = 140e6
    result = analyze(records, masks, CONFIGS)
    assert not result['geometry_magnitude_gate_pass']


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'nan', 'negative', 'component', 'mask', 'pair'])
def test_rejects_corrupt_inputs(mutation):
    masks, records = fixture()
    if mutation == 'missing':
        records.pop()
    elif mutation == 'duplicate':
        records[-1] = copy.deepcopy(records[0])
    elif mutation == 'nan':
        records[0]['service_ns'] = float('nan')
    elif mutation == 'negative':
        records[0]['normalization_ns'] = -1
    elif mutation == 'component':
        records[0]['normalization_ns'] = 1e12
    elif mutation == 'mask':
        records[0]['mask_id'] = 'unexpected'
    else:
        masks[0]['member'] = 'fragmented'
    with pytest.raises(ValueError):
        analyze(records, masks, CONFIGS)


def authenticated_fixture(tmp_path):
    bundle, run = tmp_path/'bundle', tmp_path/'run'
    bundle.mkdir(); run.mkdir()
    def sha(data):
        return hashlib.sha256(data).hexdigest()
    def encoded(value):
        return json.dumps(value, sort_keys=True).encode()
    masks, records = fixture()
    rows = ['1000000000']+['0000000000']*5
    schedule = [dict(session=0, cell='width24', repeat=r['repeat'], mask_id=r['mask_id'], rows=rows,
                     configuration='frozen_choice' if r['configuration'] == 'chooser' else r['configuration'],
                     kind='evaluation') for r in records if r['session'] == 0]
    profile = {mode: {'coefficients': [1., 1., 1.],
                     'costs': {f'{kind}:{width}': {'base_ms': 1.+width/64,
                                                  'commit_tile_ms': 1.}
                               for kind in ('top', 'middle', 'bottom') for width in WIDTHS}}
               for mode in ('whole', 'selected')}
    compiled = {'score_sha256': 'score', 'serialized_score_bytes': 10,
                'tables': {'test': {'table_sha256': 'table', 'cost_sha256': 'cost', 'serialized_table_bytes': 1}}}
    method = {'practical_gates': PRACTICAL_GATES, 'bootstrap': BOOTSTRAP_CONTRACT,
              'candidate_order': list(CONFIGS), 'profiles': {'width24': profile},
              'compiled': {'width24': compiled}, 'materialization_models': {}}
    (bundle/'method_freeze.json').write_bytes(encoded(method))
    frozen = {'masks': masks, 'schedule': schedule, 'profiles': {'width24': profile},
              'compiled': {'width24': compiled}, 'materialization_models': {},
              'method_freeze_sha256': sha((bundle/'method_freeze.json').read_bytes())}
    (bundle/'frozen.json').write_bytes(encoded(frozen))
    (bundle/'dummy.py').write_text('# fixture\n')
    pins = {p.name: sha(p.read_bytes()) for p in bundle.iterdir()}
    (bundle/'source_manifest.json').write_bytes(encoded({'files': pins}))
    output = np.zeros((720, 1280, 3), np.float32)
    array = run/'raw_fixture.npz'
    np.savez_compressed(array, output=output)
    binding = {'path': array.name, 'sha256': sha(array.read_bytes()), 'array_sha256': sha(output.tobytes())}
    events = [{'state': 'running', 'cell': 'width24', 'session': 0,
               'source_manifest_sha': sha((bundle/'source_manifest.json').read_bytes()),
               'frozen_sha256': pins['frozen.json'], 'method_freeze_sha256': pins['method_freeze.json'],
               'accepted_bundle_sha': ACCEPTED_BUNDLE_SHA}]
    clock = 0
    def control(stage, active=True):
        nonlocal clock
        clock += 1
        helper = 'c86a1ae79a72a6955b8d0dc15c75e36e2fcbe2d332b981f4d7dde69022ecb4a7'
        owners = [{'pid': 123, 'starttime': 456, 'device': [226, 1], 'fd': 6}] if active else []
        coverage = {'schema': 1, 'scope': 'RK3576_DUAL_NODE_NPU_OWNERS_ONLY', 'complete': True,
                    'covered_devices': [[226, 1], [226, 129]], 'errors': [], 'sweeps': [owners, owners],
                    'owner_vector_sha256': sha(json.dumps(owners, sort_keys=True, separators=(',', ':')).encode())}
        stdout = json.dumps(coverage)
        snapshot = {'temperature_c': 50., 'monotonic_ns': clock, 'npu_ownership_complete': True,
                    'invariants': {'other_npu_owners': [], 'board_sha256': 'board'},
                    'dual_node_capture': {'helper_sha256': helper, 'returncode': 0, 'parse_error': None,
                                         'stderr': '', 'stdout': stdout, 'stdout_sha256': sha(stdout.encode())},
                    'npu_ownership_proof': {'helper_sha256': helper, 'owners': owners, 'coverage': coverage}}
        events.append({'state': 'control', 'stage': stage, 'snapshot': snapshot})
    control('start', False)
    events.append({'state': 'loaded'})
    control('loaded')
    prepared = copy.deepcopy(compiled)
    prepared['tables']['test']['preparation_ns'] = 1.
    events.append({'state': 'preparation', **prepared, 'preparation_ns': 2.,
                   'profile_sha256': sha(json.dumps(profile, sort_keys=True, separators=(',', ':')).encode())})
    control('prepared')
    def sample(request, metrics=None):
        request = dict(request)
        request.setdefault('session', 0); request.setdefault('cell', 'width24')
        control('before_'+request['kind'])
        events.append(dict(binding, state='sample', request=request,
                           metrics=metrics or {'passed': True}))
        control('after_'+request['kind'])
    sample({'kind': 'reference'})
    for y, width, right in itertools.product((0, 300), WIDTHS, (False, True)):
        job = {'x': 640-width if right else 0, 'y': y, 'width': width, 'commits': [y//60*10+(9 if right else 0)]}
        for mode in ('whole', 'selected'):
            sample({'kind': 'qualification', 'job': job, 'reconstruction': mode})
    for req, record in zip(schedule, [r for r in records if r['session'] == 0], strict=True):
        chosen = 'exact:whole' if req['configuration'] == 'frozen_choice' else req['configuration']
        policy, mode = chosen.split(':')
        metrics = {key: 0 for key in STAGES}
        metrics.update(passed=True, service_ns=record['service_ns'], normalization_ns=20e6,
                       planning_ns=2e6, decision_ns=1e6, chosen_policy=policy, chosen_reconstruction=mode,
                       calls=1, per_job=[{'x': 0, 'y': 0, 'width': 64, 'commits': [0]}],
                       frozen_calibrated_job_cost_ms=3. if req['configuration'] == 'frozen_choice' else None)
        sample(req, metrics)
    control('closed', False)
    events.append({'state': 'complete', 'confirmation': True, 'observations': 416, 'qualifications': 48, 'references': 1})
    return bundle, run, events


def save_journal(run, events):
    (run/'journal.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))


def test_authenticates_full_cell(tmp_path):
    bundle, run, events = authenticated_fixture(tmp_path)
    save_journal(run, events)
    result = authenticate_run(run, bundle)
    assert len(result['records']) == 416
    assert result['fixed_reconstruction_pairs'] == 96
    assert result['unique_raw_arrays'] == 1
    assert result['recorded_control_audit']['snapshots'] == 934
    assert result['recorded_control_audit']['recorded_pid_starttime'] == [(123, 456)]


@pytest.mark.parametrize('mutation', ['order', 'source', 'profile', 'commits', 'choice', 'hash', 'qualification', 'timer'])
def test_authentication_rejects_drift(tmp_path, mutation):
    bundle, run, events = authenticated_fixture(tmp_path)
    evaluations = [e for e in events if e.get('request', {}).get('kind') == 'evaluation']
    if mutation == 'order':
        a, b = events.index(evaluations[0]), events.index(evaluations[1])
        events[a], events[b] = events[b], events[a]
    elif mutation == 'source':
        (bundle/'dummy.py').write_text('# changed\n')
    elif mutation == 'profile':
        next(e for e in events if e['state'] == 'preparation')['profile_sha256'] = 'wrong'
    elif mutation == 'commits':
        evaluations[0]['metrics']['per_job'][0]['commits'] = [1]
    elif mutation == 'choice':
        evaluations[0]['metrics']['chosen_policy'] = 'grouped'
    elif mutation == 'hash':
        evaluations[0]['array_sha256'] = 'wrong'
    elif mutation == 'qualification':
        next(e for e in events if e.get('request', {}).get('kind') == 'qualification')['request']['job']['width'] = 65
    else:
        evaluations[0]['metrics']['decision_ns'] = 1e12
    save_journal(run, events)
    with pytest.raises(ValueError):
        authenticate_run(run, bundle)


def test_requires_requested_external_manifest_pin(tmp_path):
    bundle, run, events = authenticated_fixture(tmp_path)
    save_journal(run, events)
    correct = hashlib.sha256((bundle/'source_manifest.json').read_bytes()).hexdigest()
    authenticate_run(run, bundle, expected_manifest_sha=correct)
    with pytest.raises(ValueError, match='accepted frozen manifest'):
        authenticate_run(run, bundle, expected_manifest_sha=ACCEPTED_SOURCE_SHA)


@pytest.mark.parametrize('mutation', ['winner', 'score', 'fixed_jobs', 'chooser_jobs', 'fixed_score'])
def test_recomputes_decisions_not_merely_recorded_coverage(tmp_path, mutation):
    bundle, run, events = authenticated_fixture(tmp_path)
    evaluations = [e for e in events if e.get('request', {}).get('kind') == 'evaluation']
    chosen = next(e for e in evaluations if e['request']['configuration'] == 'frozen_choice')
    fixed = next(e for e in evaluations if e['request']['configuration'] == 'lookup_partition:whole')
    if mutation == 'winner':
        # Same job, same output, same predicted cost: violates the frozen tie order.
        chosen['metrics']['chosen_reconstruction'] = 'selected'
    elif mutation == 'score':
        chosen['metrics']['frozen_calibrated_job_cost_ms'] = 3.001
    elif mutation == 'fixed_jobs':
        # Still owns exactly the same selected tile and is a valid wider shape.
        fixed['metrics']['per_job'][0]['width'] = 128
    elif mutation == 'chooser_jobs':
        # Make both the chooser and its fixed counterpart agree on wrong jobs.
        for row in evaluations:
            if row['request']['configuration'] in ('exact:whole', 'exact:selected', 'frozen_choice'):
                row['metrics']['per_job'][0]['width'] = 128
    else:
        fixed['metrics']['frozen_calibrated_job_cost_ms'] = 3.
    save_journal(run, events)
    with pytest.raises(ValueError):
        authenticate_run(run, bundle)


@pytest.mark.parametrize('mutation', ['binding', 'gate', 'bootstrap', 'tie_order', 'profile_binding'])
def test_rejects_self_consistent_but_different_method(tmp_path, mutation):
    bundle, run, events = authenticated_fixture(tmp_path)
    method = json.loads((bundle/'method_freeze.json').read_bytes())
    frozen = json.loads((bundle/'frozen.json').read_bytes())
    if mutation == 'gate':
        method['practical_gates']['chooser_mean_gain_fraction'] = .01
    elif mutation == 'bootstrap':
        method['bootstrap']['draws'] = 200
    elif mutation == 'tie_order':
        method['candidate_order'].reverse()
    elif mutation == 'profile_binding':
        method['profiles']['width24']['whole']['coefficients'][0] = 2.
    raw = json.dumps(method, sort_keys=True).encode()
    (bundle/'method_freeze.json').write_bytes(raw)
    frozen['method_freeze_sha256'] = 'wrong' if mutation == 'binding' else hashlib.sha256(raw).hexdigest()
    (bundle/'frozen.json').write_text(json.dumps(frozen, sort_keys=True))
    manifest = json.loads((bundle/'source_manifest.json').read_bytes())
    for name in manifest['files']:
        manifest['files'][name] = hashlib.sha256((bundle/name).read_bytes()).hexdigest()
    (bundle/'source_manifest.json').write_text(json.dumps(manifest, sort_keys=True))
    events[0].update(source_manifest_sha=hashlib.sha256((bundle/'source_manifest.json').read_bytes()).hexdigest(),
                     frozen_sha256=manifest['files']['frozen.json'],
                     method_freeze_sha256=manifest['files']['method_freeze.json'])
    save_journal(run, events)
    with pytest.raises(ValueError):
        authenticate_run(run, bundle)


@pytest.mark.parametrize('mutation', ['missing_after', 'helper', 'capture', 'coverage', 'foreign',
                                    'invariants', 'temperature', 'start_temperature', 'clock'])
def test_authenticates_point_in_time_controls(tmp_path, mutation):
    bundle, run, events = authenticated_fixture(tmp_path)
    target = next(e for e in events if e.get('stage') == 'after_evaluation')
    snap = target['snapshot']
    if mutation == 'missing_after':
        events.remove(target)
    elif mutation == 'helper':
        snap['dual_node_capture']['helper_sha256'] = 'changed'
    elif mutation == 'capture':
        snap['dual_node_capture']['stdout'] += ' '
    elif mutation == 'coverage':
        snap['npu_ownership_complete'] = False
    elif mutation == 'foreign':
        snap['invariants']['other_npu_owners'] = [{'pid': 999}]
    elif mutation == 'invariants':
        snap['invariants']['board_sha256'] = 'different'
    elif mutation == 'temperature':
        snap['temperature_c'] = 80.
    elif mutation == 'start_temperature':
        next(e for e in events if e.get('stage') == 'start')['snapshot']['temperature_c'] = 60.01
    else:
        snap['monotonic_ns'] = 0
    save_journal(run, events)
    with pytest.raises(ValueError):
        authenticate_run(run, bundle)


def test_independent_jobs_and_scores_match_frozen_v3_pure_planner():
    """Only source/profile/masks, no held-out outcomes or board access."""
    from src.experiments.paper3_strengthening_v2.cached import compile_choice, CANDIDATES
    root = Path(__file__).resolve().parents[1]
    bundle = root/'results/paper3_strengthening_v2/E002_source_v3'
    assert hashlib.sha256((bundle/'source_manifest.json').read_bytes()).hexdigest() == ACCEPTED_SOURCE_SHA
    for name in ('core.py', 'cached.py'):
        assert ((bundle/'strength'/name).read_bytes() ==
                (root/'src/experiments/paper3_strengthening_v2'/name).read_bytes())
    frozen = json.loads((bundle/'frozen.json').read_bytes())
    for cell in ('width24', 'fp16'):
        calibrated = frozen['profiles'][cell]
        compiled = compile_choice(calibrated)
        assert compiled.score_sha256 == frozen['compiled'][cell]['score_sha256']
        for mask in frozen['masks']:
            expected, winner = frozen_expectations(calibrated, mask['rows'])
            policy, mode, score = compiled.decide(mask['rows'])
            assert winner == policy+':'+mode
            assert expected[winner]['score_ms'] == score
            for policy, mode in CANDIDATES:
                jobs = compiled.planner(policy, mode)(mask['rows'])
                assert tuple((j.x, j.y, j.width, j.commits) for j in jobs) == expected[policy+':'+mode]['jobs']
