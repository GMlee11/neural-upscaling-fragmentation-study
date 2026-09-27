import copy
import hashlib
import json
from pathlib import Path
import pytest
from src.report.paper3_strengthening_predictions import (
    FIXED, FULL_REFERENCE_SHA, analyze, predict, amortization, total, source_binding,
    bind_numerical_reports,
)
from src.report.paper3_strengthening_confirmation import ACCEPTED_SOURCE_SHA, frozen_expectations


@pytest.fixture
def data():
    profiles = {m: {'costs': {'top:128': {'base_ms': 2, 'commit_tile_ms': .25}}}
                for m in ('whole','selected')}
    coefficients = {'call_ms': .1, 'million_elements_ms': 10}
    samples = []
    for session in (0,1):
        for repeat in (0,1):
            for mask in range(16):
                for config in (*FIXED,'frozen_choice'):
                    chosen = 'exact:selected' if config == 'frozen_choice' else config
                    policy,mode = chosen.split(':')
                    service = 9 if config == 'frozen_choice' else 10 if config == 'exact:selected' else 12
                    samples.append({'request': {'kind':'evaluation','session':session,'repeat':repeat,
                                                'mask_id':f'm{mask}','configuration':config},
                                    'metrics': {'chosen_policy':policy,'chosen_reconstruction':mode,
                                                'per_job':[{'y':0,'width':128,'commits':[0,1],'output_elements':100000,
                                                            'extraction_ns':500000,'inference_ns':1000000,
                                                            'normalization_ns':1000000,'scatter_ns':100000,
                                                            'job_ns':2700000}],
                                                'extraction_ns':500000,'inference_ns':1000000,
                                                'normalization_ns':1000000,'scatter_ns':100000,
                                                'service_ns':service*1000000,'decision_ns':1000}})
    return samples,profiles,coefficients


def test_correct_prediction_boundaries(data):
    samples,profiles,coefficients=data
    job=samples[0]['metrics']['per_job']
    whole=predict(job,'whole',profiles['whole'],coefficients)
    selected=predict(job,'selected',profiles['selected'],coefficients)
    assert whole['predicted_job_components_ms']==2.5
    assert whole['predicted_normalization_ms']==pytest.approx(1.1)
    assert selected['normalization_calls']==2
    assert selected['materialized_elements']==92160
    assert selected['predicted_normalization_ms']==pytest.approx(1.1216)


def test_descriptive_report_and_amortization(data):
    samples,profiles,coefficients=data
    result=analyze(samples,profiles,coefficients,[20000,22000],5000)
    assert result['best_fixed']=='exact:selected'
    assert result['chooser_mean_ms']==9
    assert result['conditional_amortization']['total_preparation_requests']==21000
    assert result['conditional_amortization']['preparation_plus_calibration_service_requests']==26000
    assert len(result['wrong_choices'])==16
    assert all(r['measured_choice_penalty_ms']==0 for r in result['wrong_choices'])
    assert len(result['normalization_pairs'])==192
    assert result['job_cost_prediction']['bias_ms']==pytest.approx(-.1)


def test_incomplete_matrix_rejected(data):
    samples,profiles,coefficients=data
    with pytest.raises(ValueError,match='incomplete'):
        analyze(samples[:-1],profiles,coefficients,[20,21],5)


def test_duplicate_rejected(data):
    samples,profiles,coefficients=data
    with pytest.raises(ValueError,match='duplicate'):
        analyze(samples+[samples[0]],profiles,coefficients,[20,21],5)


def test_changed_choice_rejected(data):
    samples,profiles,coefficients=data
    next(s for s in samples if s['request']['configuration']=='frozen_choice')['metrics']['chosen_policy']='grouped'
    with pytest.raises(ValueError,match='changed'):
        analyze(samples,profiles,coefficients,[20,21],5)


@pytest.mark.parametrize('saving',[0,-1])
def test_no_amortization_without_positive_saving(saving):
    assert amortization(20000,saving) is None


@pytest.mark.parametrize('value',[-1,float('nan'),float('inf'),True,'2'])
def test_invalid_nonnegative_cost(value):
    with pytest.raises(ValueError):total([value])


@pytest.mark.parametrize('mutation', ['service_nan', 'zero_service', 'decision_negative', 'component_overflow',
                                    'job_component', 'job_elapsed', 'setup_nan', 'setup_count', 'calibration_negative'])
def test_rejects_invalid_measured_boundaries(data, mutation):
    samples,profiles,coefficients=data
    met=samples[0]['metrics'];preparation=[20,21];calibration=5
    if mutation=='service_nan':met['service_ns']=float('nan')
    elif mutation=='zero_service':met['service_ns']=0
    elif mutation=='decision_negative':met['decision_ns']=-1
    elif mutation=='component_overflow':met['inference_ns']=1e20
    elif mutation=='job_component':met['per_job'][0]['inference_ns']+=1
    elif mutation=='job_elapsed':met['per_job'][0]['job_ns']=1
    elif mutation=='setup_nan':preparation=[float('nan'),21]
    elif mutation=='setup_count':preparation=[20]
    else:calibration=-1
    with pytest.raises(ValueError):analyze(samples,profiles,coefficients,preparation,calibration)


def test_scored_components_use_frozen_rowwise_accumulation():
    bundle=Path(__file__).resolve().parents[1]/'results/paper3_strengthening_v2/E002_source_v3'
    frozen=json.loads((bundle/'frozen.json').read_bytes())
    # Profiles and reserved geometries only; no measured E002 records.
    for cell in ('width24','fp16'):
        for mask in frozen['masks']:
            expected,_=frozen_expectations(frozen['profiles'][cell],mask['rows'])
            for config,value in expected.items():
                _,mode=config.split(':')
                jobs=[dict(x=x,y=y,width=w,commits=list(c),output_elements=w*60*12)
                      for x,y,w,c in value['jobs']]
                predicted=predict(jobs,mode,frozen['profiles'][cell][mode],frozen['materialization_models'][cell])
                assert predicted['predicted_job_components_ms']==value['score_ms']


def audit_fixture():
    bundle=Path(__file__).resolve().parents[1]/'results/paper3_strengthening_v2/E002_source_v3'
    frozen_sha=hashlib.sha256((bundle/'frozen.json').read_bytes()).hexdigest()
    runs=[dict(cell=cell,session=session,journal_sha256=f'{cell}-{session}',
               source_manifest_sha256=ACCEPTED_SOURCE_SHA,frozen_sha256=frozen_sha)
          for cell in ('width24','fp16') for session in (0,1)]
    return bundle, {'status':'complete_saved_record_audit','runs':runs}


def numerical_fixture(authenticated):
    return [dict(status='retained_pixel_replay_passed',confirmation=True,sample_count=465,cell=cell,
                 session=session,source_manifest_sha256=ACCEPTED_SOURCE_SHA,
                 journal_sha256=report['journal_sha256'],fixed_job_reconstruction_pairs=96,
                 fixed_job_reconstruction_pairs_byte_identical=True,
                 independent_fp16_reference_used=cell=='fp16',
                 full_reference_file_sha256=FULL_REFERENCE_SHA if cell=='fp16' else None)
            for (cell,session),report in authenticated.items()]


def test_source_and_numerical_report_bindings():
    bundle,audit=audit_fixture()
    frozen,_,_,authenticated=source_binding(bundle,audit)
    assert len(frozen['masks'])==16
    bindings=bind_numerical_reports(numerical_fixture(authenticated),authenticated)
    assert set(bindings.values())=={(c,s) for c in ('width24','fp16') for s in (0,1)}


@pytest.mark.parametrize('mutation',['source','freeze','duplicate','missing','status','journal'])
def test_rejects_inconsistent_completed_audit(mutation):
    bundle,audit=audit_fixture()
    if mutation=='source':audit['runs'][0]['source_manifest_sha256']='wrong'
    elif mutation=='freeze':audit['runs'][0]['frozen_sha256']='wrong'
    elif mutation=='duplicate':audit['runs'][0]=copy.deepcopy(audit['runs'][1])
    elif mutation=='missing':audit['runs'].pop()
    elif mutation=='status':audit['status']='incomplete'
    else:audit['runs'][0]['journal_sha256']=audit['runs'][1]['journal_sha256']
    with pytest.raises(ValueError):source_binding(bundle,audit)


@pytest.mark.parametrize('mutation',['duplicate','missing','status','count','reference','pair','journal','confirmation','source','session'])
def test_rejects_unbound_numerical_reports(mutation):
    bundle,audit=audit_fixture()
    _,_,_,authenticated=source_binding(bundle,audit)
    reports=numerical_fixture(authenticated)
    if mutation=='duplicate':reports[0]=copy.deepcopy(reports[1])
    elif mutation=='missing':reports.pop()
    elif mutation=='status':reports[0]['status']='failed'
    elif mutation=='count':reports[0]['sample_count']=445
    elif mutation=='reference':reports[2]['full_reference_file_sha256']='wrong'
    elif mutation=='pair':reports[0]['fixed_job_reconstruction_pairs_byte_identical']=False
    elif mutation=='journal':reports[0]['journal_sha256']='unknown'
    elif mutation=='source':reports[0]['source_manifest_sha256']='wrong'
    elif mutation=='session':reports[0]['session']=1
    else:reports[0]['confirmation']=False
    with pytest.raises(ValueError):bind_numerical_reports(reports,authenticated)
