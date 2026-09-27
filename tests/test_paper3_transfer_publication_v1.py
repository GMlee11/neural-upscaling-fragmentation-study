import copy
import json
from pathlib import Path
import pytest
from src.report.paper3_transfer_publication_v1 import summarize, latex, INPUT

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def payload(): return json.loads((ROOT/INPUT).read_bytes())

def test_transfer_replay(payload):
    result=summarize(payload)
    assert result == json.loads((ROOT/'results/tables/paper3_transfer_publication_v1.json').read_bytes())
    assert latex(result)==(ROOT/'docs/paper/generated/paper3_transfer_publication_v1.tex').read_text()
    assert result['samples']==5120
    assert all(min(c['session_reductions_percent'])>0 for c in result['cells'])
    assert all(c['policies']['calibrated']['regressions_vs_best_fixed']>0 for c in result['cells'])

@pytest.mark.parametrize('kind',['time','mode','count','schedule','coefficient','audit','fit'])
def test_mutations_fail(payload,kind):
    p=copy.deepcopy(payload); c=p['cells'][0]; r=c['sessions'][0]['records'][0]
    if kind=='time': r['service_ms']+=100
    elif kind=='mode': r['chosen_mode']='row_spans'
    elif kind=='count': c['sessions'][0]['records'].pop()
    elif kind=='schedule': r['rows'][0]='0000000000'
    elif kind=='coefficient': p['donor']['coefficients'][0]+=1
    elif kind=='audit': c['sessions'][0]['status']='failed'
    else: p['donor']['additional_target_calibration_requests']=1
    with pytest.raises(ValueError): summarize(p)
