"""E002 timed conventional choice and complete-output validation boundary."""
import time
import numpy as np

from .service import execute
from src.experiments.paper3_dispatch_successor_v1.development import output_integrity


def evaluate(req, backend):
    rows=tuple(req['rows'])
    start=time.perf_counter_ns()
    if req['policy']=='frozen_choice':
        policy,mode,score=backend.choice.decide(rows)
    else:
        policy,mode,score=req['policy'],req['reconstruction'],None
    planner=backend.choice.planner(policy,mode)
    decision_ns=time.perf_counter_ns()-start
    output,metrics=execute(backend.fixture,rows,planner,backend,mode)
    # Measured once: execution already includes actual planning and fallback.
    metrics['service_ns']=time.perf_counter_ns()-start
    metrics['planning_ns']+=decision_ns
    metrics.update(decision_ns=decision_ns,chosen_policy=policy,chosen_reconstruction=mode,
                   frozen_calibrated_job_cost_ms=score)
    return validate_completed(output,metrics,rows,backend)


def validate_completed(output,metrics,rows,backend):
    # A completed but invalid array must reach the parent for durable retention.
    # Native execution failure before producing output still uses worker error.
    try:
        if output.shape!=(720,1280,3) or not np.isfinite(output).all():
            raise ValueError('invalid completed output')
        metrics['integrity']=output_integrity(output,backend.strong_reference,backend.baseline,rows)
        mask=np.array([[bit=='1' for bit in row] for row in rows])
        errors=[]
        for r,c in zip(*np.nonzero(mask)):
            area=np.s_[r*120:(r+1)*120,c*128:(c+1)*128]
            errors.append(float(np.abs(output[area].astype(np.float64)-backend.strong_reference[area]).mean()))
        metrics.update(max_tile_mae=max(errors),per_tile_mae=errors)
        if backend.cell=='fp16':
            selected=np.repeat(np.repeat(mask,120,axis=0),128,axis=1)
            error=float(np.abs(output[selected].astype(np.float64)-backend.reference[selected]).mean())
            metrics['selected_full_reference_mae']=error
            if error>.001:raise ValueError('FP16 independent reference threshold')
        metrics['passed']=True
    except Exception as error:
        metrics.update(passed=False,validation_failure=repr(error))
    return output,metrics
