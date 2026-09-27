"""Render all frozen E002 alternatives; no result-dependent row selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

POLICIES = ('exact', 'row_spans', 'grouped', 'per_row', 'affine_partition', 'lookup_partition')
LABELS = ('Individual tiles', 'Row spans', 'Contiguous groups', 'Per-row choice',
          'Affine partition', 'Profiled partition')
FAMILIES = ('sparse_all_rows', 'sparse_three_rows', 'dense_all_rows', 'dense_four_rows')
FAMILY_LABELS = (r'$6\times2$', r'$3\times4$', r'$6\times6$', r'$4\times9$')


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('nonfinite or invalid publication value')
    return float(value)


def build(report):
    if report['status'] != 'complete_saved_record_audit' or set(report['graphs']) != {'width24', 'fp16'}:
        raise ValueError('requires complete two-graph saved-record audit')
    configs = {f'{p}:{m}' for p in POLICIES for m in ('whole', 'selected')} | {'frozen_choice'}
    for graph in report['graphs'].values():
        if (graph['status'] != 'complete_practical_analysis' or graph['evaluation_rows'] != 832 or
                set(graph['configurations']) != configs or set(graph['geometry_families']) != set(FAMILIES)):
            raise ValueError('incomplete frozen comparison')
    service = []
    for policy, label in zip(POLICIES, LABELS):
        values = [number(report['graphs'][g]['configurations'][f'{policy}:{m}']['mean_service_ms'])
                  for g in ('width24', 'fp16') for m in ('whole', 'selected')]
        if min(values) <= 0:
            raise ValueError('nonpositive service time')
        service.append({'policy': policy, 'label': label, 'means_ms': values})
    choices = {g: number(report['graphs'][g]['configurations']['frozen_choice']['mean_service_ms'])
               for g in ('width24', 'fp16')}
    geometry = []
    for graph in ('width24', 'fp16'):
        for family, label in zip(FAMILIES, FAMILY_LABELS):
            item = report['graphs'][graph]['geometry_families'][family]
            sessions = item['sessions']
            if set(sessions) != {'0', '1'} or any(len(s['pairs']) != 2 for s in sessions.values()):
                raise ValueError('incomplete paired geometry')
            whole, selected = [mean(number(p['service'][mode]) for s in sessions.values()
                                     for p in s['pairs'].values()) for mode in ('whole', 'selected')]
            interaction = number(item['pooled_service_intervention_ms'])
            if abs(whole-selected-interaction) > 1e-8:
                raise ValueError('inconsistent geometry interaction')
            geometry.append({'graph': graph, 'family': family, 'label': label,
                             'whole_penalty_ms': whole, 'selected_penalty_ms': selected,
                             'interaction_ms': interaction,
                             'same_sign': bool(item['same_sign_in_both_sessions']),
                             'practical_gate': bool(item['magnitude_gate_pass'])})
    return {'service': service, 'chooser_ms': choices, 'geometry': geometry,
            'scope': 'All 12 fixed configurations plus actual chooser; 16 masks, two sessions, two repeats. Numerical audit is separate.'}


def render_service(data):
    lines = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'& \multicolumn{2}{c}{Width-24} & \multicolumn{2}{c}{FP16} \\',
             r'Organization & Whole & Selected & Whole & Selected \\', r'\midrule']
    for row in data['service']:
        lines.append(row['label']+' & '+' & '.join(f'{v:.3f}' for v in row['means_ms'])+r' \\')
    choices = data['chooser_ms']
    lines += [r'\midrule', 'Frozen choice & '+r'\multicolumn{2}{c}{'+f"{choices['width24']:.3f}"+
              '} & '+r'\multicolumn{2}{c}{'+f"{choices['fp16']:.3f}"+r'} \\',
              r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def render_geometry(data):
    lines = [r'\begin{tabular}{@{}llrrrcc@{}}', r'\toprule',
             r'Graph & Rows$\times$tiles/row & Whole penalty & Selected penalty & Interaction & Stable sign & Gate \\',
             r'\midrule']
    for row in data['geometry']:
        label = 'Width-24' if row['graph'] == 'width24' else 'FP16'
        values = ' & '.join(f'{row[k]:+.3f}' for k in ('whole_penalty_ms', 'selected_penalty_ms', 'interaction_ms'))
        lines.append(label+' & '+row['label']+' & '+values+' & '+('Yes' if row['same_sign'] else 'No')+
                     ' & '+('Pass' if row['practical_gate'] else 'Fail')+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    data = build(json.loads(raw))
    data['saved_record_audit_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {'paper3_strengthening_service_v1.tex': render_service(data),
               'paper3_strengthening_geometry_v1.tex': render_geometry(data),
               'paper3_strengthening_publication_v1.json': json.dumps(data, indent=2, allow_nan=False)+'\n'}
    for name, text in outputs.items():
        (args.output_directory/name).write_text(text, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
