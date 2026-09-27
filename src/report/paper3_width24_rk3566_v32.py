"""Deterministic offline projection and paired analysis of the frozen V32 capture."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SHA = "1b3236b61485e0e5a2d72c9c96233fd1deaa2a11882ee33c0d823eafa025c98d"
CAPTURE_BYTES = 4595914
CAPTURE_TAIL = "1d0f3ea791ab04408eefa5285c0732767e627760592415e0cf7841ee61da142c"
PUBLIC = "results/manifests/paper3_width24_int8_rk3566_public_v32.json"
ANALYSIS = "results/tables/paper3_width24_int8_rk3566_analysis_v32.json"


def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def project():
    # Historical-only path: validates raw provenance, never constructs a backend.
    from src.deployment import audit_paper3_width24_qualification_v32 as audit
    p=audit.p;qualification=audit.audit();path=p.OUT/"capture.jsonl"
    if path.stat().st_size!=CAPTURE_BYTES or p.sha(path)!=CAPTURE_SHA:
        raise ValueError("capture byte identity mismatch")
    _,rows=p.read_journal(path)
    if rows[-1]["sha256"]!=CAPTURE_TAIL or rows[-1]["payload"]!={"status":"CAPTURE_COMPLETE","samples":60,"warmups":16,"catalog_sha256":p.CATALOG_SHA}:
        raise ValueError("capture terminal contract")
    frozen=p.read(p.FREEZE);schedule=frozen["schedule"]
    expected=["claimed","offline_parent","qualification_parent","controls_start","warmups_complete"]
    for _ in range(5):expected += ["block_start"]+[x for _ in range(12) for x in ("sample_metrics","sample")]+["cross_mode"]
    expected += ["controls_end","terminal_success"]
    if [r["event"] for r in rows]!=expected:raise ValueError("capture event sequence")
    if rows[0]["payload"]!={"action":"capture","experiment_id":p.ID,"freeze_sha256":p.sha(p.FREEZE)}:raise ValueError("capture claim")
    if rows[2]["payload"]!=qualification["qualification_parent"]:raise ValueError("qualification parent")
    _,qrows=p.read_journal(p.OUT/"qualification.jsonl")
    if rows[1]["payload"]!=qrows[1]["payload"]:raise ValueError("offline parent")
    if rows[4]["payload"]!={"count":16}:raise ValueError("warmup count")
    samples=[r["payload"] for r in rows if r["event"]=="sample"]
    metrics=[r["payload"] for r in rows if r["event"]=="sample_metrics"]
    initial=rows[3]["payload"]
    controls=[r["payload"] for r in rows if r["event"] in ("controls_start","block_start","controls_end")]+[r["controls"] for r in samples]
    for c in controls:p.enforce_control_snapshot(c,initial)
    for r in rows:
        if r["event"] in ("controls_start","block_start") and r["payload"]["temperature_c"]>65:raise ValueError("cooldown start gate")
    expected_checksums={(r["payload"]["mask_id"],r["payload"]["mode"]):r["payload"]["selected_sha256"] for r in qrows if r["event"]=="numerical"}
    checks=[r["payload"] for r in rows if r["event"]=="cross_mode"]
    for check in checks:
        if [v["mask_id"] for v in check]!=sorted({s["mask_id"] for s in samples}):raise ValueError("cross mode mask keys")
        for c in check:
            if not 0<=c["cross_mode_mae"]<=.001 or not 0<=c["psnr_delta_db"]<=.02:raise ValueError("cross mode gate")
    selected=[]
    for entry,s,metric in zip(schedule,samples,metrics,strict=True):
        if any(s[k]!=v or metric[k]!=v for k,v in entry.items()):raise ValueError("schedule drift")
        if metric["stats"]!=s["stats"]:raise ValueError("sample metrics mismatch")
        p.validate_timings(s["stats"])
        if s["unselected_change"]!=0 or s["selected_sha256"]!=expected_checksums[(s["mask_id"],s["mode"])]:raise ValueError("sample integrity")
        selected.append({**{k:v for k,v in s.items() if k!="controls"},"post_sample_temperature_c":s["controls"]["temperature_c"]})
    return {"schema_version":1,"experiment_id":p.ID,"status":"FROZEN_CAPTURE_AUDITED_PUBLIC_PROJECTION",
        "raw_capture":{"path":path.relative_to(ROOT).as_posix(),"bytes":CAPTURE_BYTES,"sha256":CAPTURE_SHA,"tail_sha256":CAPTURE_TAIL,"events":137},
        "qualification_parent":qualification["qualification_parent"],"source_freeze_sha256":p.sha(p.FREEZE),"catalog_sha256":p.CATALOG_SHA,
        "schedule":schedule,"schedule_sha256":frozen["schedule_sha256"],"samples":selected,"cross_mode":checks,
        "controls":{"snapshots":len(controls),"temperature_min_c":min(c["temperature_c"] for c in controls),"temperature_max_c":max(c["temperature_c"] for c in controls),
            "identities":{k:initial[k] for k in p.CONTROL_IDENTITIES if k!="namespace"},"all_identity_rechecks_pass":True,
            "whole_system_exclusive_visibility":False,"raw_process_records_excluded":True},
        "integrity":{"all_gates_pass":True,"selected_checksums_match_qualification":True,"unselected_change":0,
            "max_cross_mode_mae":max(c["cross_mode_mae"] for r in checks for c in r),"max_psnr_delta_db":max(c["psnr_delta_db"] for r in checks for c in r)},
        "timing_samples":60,"warmups":16,"provenance_note":"Raw journal retained in historical repository; projection excludes process/FD operator detail, not measurements. Self NPU-FD gate is code-enforced, not an independent recorded self-FD transcript."}


def analyze(value):
    if value["raw_capture"]["sha256"]!=CAPTURE_SHA or digest(value["schedule"])!=value["schedule_sha256"]:raise ValueError("projection provenance")
    samples=value["samples"]
    if len(samples)!=60 or len(value["schedule"])!=60:raise ValueError("sample count")
    for s,e in zip(samples,value["schedule"],strict=True):
        if any(s[k]!=v for k,v in e.items()):raise ValueError("schedule mismatch")
        if s["unselected_change"]!=0:raise ValueError("unselected change")
        for key in ("compute_pipeline_ms","npu_ms","reconstruction_ms","dispatch_planning_ms","halo_extraction_ms","bicubic_ms"):
            if not math.isfinite(s["stats"][key]) or s["stats"][key]<=0:raise ValueError("nonpositive timing")
    pairs=[]
    for offset in range(0,60,2):
        left,right=samples[offset:offset+2]
        if any(left[k]!=right[k] for k in ("block_index","pair_id","dispatch_mode")):raise ValueError("pair adjacency")
        pair={s["geometry"]:s for s in (left,right)};c,f=pair["compact"],pair["maximally_fragmented"]
        row={k:c[k] for k in ("block_index","density_percent","dispatch_mode")}
        row.update(fragmented_second=f["execution_order"]>c["execution_order"],post_temperature_gap_c=f["post_sample_temperature_c"]-c["post_sample_temperature_c"])
        for key in ("compute_pipeline_ms","npu_ms","reconstruction_ms","dispatch_planning_ms","halo_extraction_ms","bicubic_ms","npu_calls","npu_input_bytes","npu_output_bytes","dispatched_core_pixels","selected_core_pixels"):
            a,b=c["stats"][key],f["stats"][key]
            row[key]={"compact":a,"fragmented":b,"penalty_percent":100*(b-a)/a}
        row["executed_gmac"]={"compact":c["executed_gmac"],"fragmented":f["executed_gmac"]}
        pairs.append(row)
    summaries=[]
    for den in (20,40,60):
        for mode in ("exact","row_spans"):
            group=[r for r in pairs if r["density_percent"]==den and r["dispatch_mode"]==mode]
            if len(group)!=5:raise ValueError("pair count")
            penalties=[r["compute_pipeline_ms"]["penalty_percent"] for r in group]
            summaries.append({"density_percent":den,"dispatch_mode":mode,"paired_blocks":5,"service_penalties_percent":penalties,
                "median_service_penalty_percent":statistics.median(penalties),"range_percent":[min(penalties),max(penalties)],
                "positive_pairs":sum(x>0 for x in penalties),"fragmented_second_count":sum(r["fragmented_second"] for r in group),
                "median_npu_penalty_percent":statistics.median(r["npu_ms"]["penalty_percent"] for r in group),
                "selected_counts_equal":all(r["selected_core_pixels"]["compact"]==r["selected_core_pixels"]["fragmented"] for r in group),
                "launch_counts_equal":all(r["npu_calls"]["compact"]==r["npu_calls"]["fragmented"] for r in group),
                "median_input_byte_inflation_percent":statistics.median(r["npu_input_bytes"]["penalty_percent"] for r in group),
                "median_output_byte_inflation_percent":statistics.median(r["npu_output_bytes"]["penalty_percent"] for r in group)})
    by={(s["density_percent"],s["dispatch_mode"]):s for s in summaries}
    gates={"row_span_positive_all_densities":all(by[d,"row_spans"]["median_service_penalty_percent"]>0 for d in (20,40,60)),
        "row_span_ge5_at_two_densities":sum(by[d,"row_spans"]["median_service_penalty_percent"]>=5 for d in (20,40,60))>=2,
        "exact_median_abs_le5_all_densities":all(abs(by[d,"exact"]["median_service_penalty_percent"])<=5 for d in (20,40,60)),
        "exact_structural_control":all(r["selected_counts_equal"] and r["launch_counts_equal"] and r["median_input_byte_inflation_percent"]==0 and r["median_output_byte_inflation_percent"]==0 for r in summaries if r["dispatch_mode"]=="exact")}
    return {"experiment_id":value["experiment_id"],"status":"FROZEN_OFFLINE_ANALYSIS_COMPLETE","source_capture_sha256":CAPTURE_SHA,
        "projection_sha256":digest(value),"samples":60,"paired_observations":30,"pairs":pairs,"summaries":summaries,
        "primary_gates":gates,"all_primary_gates_pass":all(gates.values()),"deadline_ms":33.33,
        "deadline_successes":sum(s["stats"]["compute_pipeline_ms"]<=33.33 for s in samples),
        "service_range_ms":[min(s["stats"]["compute_pipeline_ms"] for s in samples),max(s["stats"]["compute_pipeline_ms"] for s in samples)],
        "integrity":value["integrity"],"controls":value["controls"],
        "scope":"Within-backend paired ratios primary; cross-device absolute times descriptive only. Five repeats, one session, no CI/p-value. Post-sample temperatures observed, no per-sample wall-clock timestamps or manipulated thermal factor; no universal, energy, selector or real-time claim."}


def latex(result):
    lines=[r"\begin{tabular}{llrrl}",r"\toprule",r"Density & Mode & Median (\%) & NPU (\%) & Five service deltas (\%) \\",r"\midrule"]
    for s in result["summaries"]:
        values=", ".join(f"{v:.2f}" for v in s["service_penalties_percent"])
        lines.append(f"{s['density_percent']}\\% & {s['dispatch_mode'].replace('_',' ')} & {s['median_service_penalty_percent']:.3f} & {s['median_npu_penalty_percent']:.3f} & {values} \\\\")
    return "\n".join(lines+[r"\bottomrule",r"\end{tabular}"])+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-output",type=Path)
    parser.add_argument("--input",type=Path,default=ROOT/PUBLIC)
    parser.add_argument("--json-output",type=Path)
    parser.add_argument("--latex-output",type=Path)
    args=parser.parse_args()
    if args.project_output:
        value=project();args.project_output.parent.mkdir(parents=True,exist_ok=True)
        with args.project_output.open("xb") as f:f.write(canonical(value))
    else:value=json.loads(args.input.read_bytes())
    result=analyze(value)
    for path,data in ((args.json_output,canonical(result)),(args.latex_output,latex(result).encode())):
        if path:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("xb") as f:f.write(data)
    print(json.dumps({k:v for k,v in result.items() if k not in ("pairs","controls")},sort_keys=True))


if __name__=="__main__":main()
