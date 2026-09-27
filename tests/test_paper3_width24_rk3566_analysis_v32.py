"""Compact reviewer replay; never constructs a runtime or opens traces."""
import copy
import json
from pathlib import Path
import pytest
from src.report import paper3_width24_rk3566_v32 as m


def test_frozen_compact_replay_and_threshold_boundary():
    value=json.loads((m.ROOT/m.PUBLIC).read_bytes())
    result=m.analyze(value)
    assert m.canonical(result)==(m.ROOT/m.ANALYSIS).read_bytes()
    assert m.latex(result)==(m.ROOT/'docs/paper/generated/paper3_width24_int8_rk3566_v32.tex').read_text()
    assert result['all_primary_gates_pass'] and result['deadline_successes']==0
    rows=[s for s in result['summaries'] if s['dispatch_mode']=='row_spans']
    assert [s['median_service_penalty_percent'] for s in rows]==pytest.approx([181.93677651149795,102.90679732479289,4.144967332169294])
    assert sum(s['median_service_penalty_percent']>=5 for s in rows)==2
    assert sum(s['positive_pairs'] for s in rows)==15
    assert not any('processes' in s or 'controls' in s for s in value['samples'])


@pytest.mark.parametrize('change',['missing','schedule','pair','nonfinite','unselected'])
def test_compact_tamper_rejects(change):
    value=json.loads((m.ROOT/m.PUBLIC).read_bytes())
    if change=='missing':value['samples'].pop()
    elif change=='schedule':value['samples'][0]['execution_order']=99
    elif change=='pair':value['samples'][0]['geometry']='compact'
    elif change=='nonfinite':value['samples'][0]['stats']['compute_pipeline_ms']=float('nan')
    else:value['samples'][0]['unselected_change']=1
    with pytest.raises((ValueError,KeyError)):m.analyze(value)


def test_compact_source_has_no_physical_execution_surface():
    import inspect
    source=inspect.getsource(m)
    assert 'Backend(' not in source and 'subprocess' not in source
    assert 'open("xb")' in source
