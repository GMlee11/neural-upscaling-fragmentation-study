import copy
import pytest
from src.report.paper3_strengthening_publication import build, render_service, render_geometry, FAMILIES, POLICIES


def fixture():
    graphs = {}
    for graph in ('width24', 'fp16'):
        configs = {f'{p}:{m}': {'mean_service_ms': 100+i} for i, p in enumerate(POLICIES) for m in ('whole', 'selected')}
        configs['frozen_choice'] = {'mean_service_ms': 110}
        family = {'sessions': {str(s): {'pairs': {str(p): {'service': {'whole': 10, 'selected': 15}}
                                               for p in range(2)}} for s in range(2)},
                  'pooled_service_intervention_ms': -5,
                  'same_sign_in_both_sessions': True, 'magnitude_gate_pass': True}
        graphs[graph] = {'status': 'complete_practical_analysis', 'evaluation_rows': 832,
                         'configurations': configs, 'geometry_families': {f: copy.deepcopy(family) for f in FAMILIES}}
    return {'status': 'complete_saved_record_audit', 'graphs': graphs}


def test_all_controls_and_negative_interactions_are_retained():
    data = build(fixture())
    assert len(data['service']) == 6 and len(data['geometry']) == 8
    assert data['chooser_ms']['width24'] == 110
    assert all(row['interaction_ms'] == -5 for row in data['geometry'])
    assert '110.000' in render_service(data)
    assert render_geometry(data).count('-5.000') == 8


@pytest.mark.parametrize('mutation', ('graph', 'count', 'configuration', 'family', 'nan', 'interaction'))
def test_invalid_summary_fails_closed(mutation):
    report = fixture()
    g = report['graphs']['width24']
    if mutation == 'graph': report['graphs'].pop('fp16')
    elif mutation == 'count': g['evaluation_rows'] = 831
    elif mutation == 'configuration': g['configurations'].pop('grouped:selected')
    elif mutation == 'family': g['geometry_families'].pop(FAMILIES[0])
    elif mutation == 'nan': g['configurations']['exact:whole']['mean_service_ms'] = float('nan')
    else: g['geometry_families'][FAMILIES[0]]['pooled_service_intervention_ms'] = 5
    with pytest.raises(ValueError): build(report)
