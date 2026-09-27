"""Equivalent full-residual and selected-only host reconstruction controls.

Cropping before PixelShuffle is a conventional optimization, not new science.
All inference jobs, outputs, clamping, requested pixels and fallback are held
fixed. Integer raw tensors are forbidden until explicit quantization support.
"""
import time
import numpy as np

from .core import verify_jobs
from src.deployment.benchmark_rk3576_sparse import make_dispatch_descriptor, extract_halo_patch, normalize_subpixel_residual, resize_bicubic
from src.deployment.paper3_progressive_zero_padding_v2 import extract_zero_halo_patch


def selected_residual(raw, x, y, width=64, height=60):
    if raw.ndim != 4 or raw.shape[0:2] != (1,12) or not np.issubdtype(raw.dtype,np.floating):
        raise ValueError('qualified float NCHW12 output required')
    part=raw[:,:,y:y+height,x:x+width]
    if part.shape != (1,12,height,width):
        raise ValueError('selected crop outside tensor')
    return normalize_subpixel_residual(part,input_width=width,input_height=height,
                                      quantization_scale=None,quantization_zero_point=None)


def execute(image,rows,planner,backend,reconstruction='whole'):
    if reconstruction not in ('whole','selected'):
        raise ValueError('unknown reconstruction control')
    start=time.perf_counter_ns()
    jobs=verify_jobs(rows,planner(rows))
    planned=time.perf_counter_ns()
    baseline=resize_bicubic(image);out=np.clip(baseline,0.,1.)
    fallback=time.perf_counter_ns()
    extract_ns=infer_ns=normalize_ns=scatter_ns=0
    per_job=[]
    for j in jobs:
        t=time.perf_counter_ns()
        desc=make_dispatch_descriptor(frame_id=0,frame_width=640,frame_height=360,
            core_x=j.x,core_y=j.y,core_width=j.width,core_height=60,
            halo=backend.halo,deadline_us=33333,priority=1.)
        if backend.boundary=='zero':
            patch,x,y=extract_zero_halo_patch(image,desc,halo=backend.halo)
        else:
            patch=extract_halo_patch(image,desc,halo=backend.halo);x=y=backend.halo
        t1=time.perf_counter_ns()
        variant=('full_width_top' if j.y==0 else 'full_width_bottom' if j.y==300 else 'full_width_middle') if backend.boundary=='zero' and j.width==640 else None
        raw=backend.infer_boundary(variant,(j.width,60),patch) if variant else backend.infer((j.width,60),patch)
        t2=time.perf_counter_ns()
        norm=0;scatter=0
        if reconstruction=='whole':
            residual=normalize_subpixel_residual(raw,input_width=patch.shape[1],input_height=patch.shape[0],quantization_scale=None,quantization_zero_point=None)
            t3=time.perf_counter_ns();norm=t3-t2
        for tile in j.commits:
            r,c=divmod(tile,10);sx=x+c*64-j.x
            if reconstruction=='selected':
                tn=time.perf_counter_ns();piece=selected_residual(raw,sx,y);norm+=time.perf_counter_ns()-tn
            else:
                piece=residual[y*2:y*2+120,sx*2:sx*2+128]
            ts=time.perf_counter_ns()
            area=np.s_[r*120:(r+1)*120,c*128:(c+1)*128]
            out[area]=np.clip(baseline[area]+piece,0.,1.)
            scatter+=time.perf_counter_ns()-ts
        end_job=time.perf_counter_ns()
        row={'x':j.x,'y':j.y,'width':j.width,'commits':list(j.commits),
             'extraction_ns':t1-t,'inference_ns':t2-t1,'normalization_ns':norm,
             'scatter_ns':scatter,'job_ns':end_job-t,
             'input_elements':int(patch.size),'output_elements':int(raw.size)}
        per_job.append(row)
        extract_ns+=t1-t;infer_ns+=t2-t1;normalize_ns+=norm;scatter_ns+=scatter
    end=time.perf_counter_ns()
    return out,{'service_ns':end-start,'planning_ns':planned-start,'fallback_ns':fallback-planned,
        'extraction_ns':extract_ns,'inference_ns':infer_ns,'normalization_ns':normalize_ns,
        'scatter_ns':scatter_ns,'calls':len(jobs),'per_job':per_job,'reconstruction':reconstruction}
