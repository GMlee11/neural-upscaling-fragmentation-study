"""Conventional grouped execution with the same output-mask semantics.

Not a new algorithm. Imported deployment functions preserve the accepted graph
boundary and subpixel arithmetic. Timed range includes planning and composition.
"""
import time
import numpy as np
from .core import selected, validate_mask
from src.deployment.benchmark_rk3576_sparse import make_dispatch_descriptor, extract_halo_patch, resize_bicubic, normalize_subpixel_residual
from src.deployment.paper3_progressive_zero_padding_v2 import extract_zero_halo_patch

def execute_jobs(image,rows,planner,backend):
    begin=time.perf_counter_ns();rows=validate_mask(rows);jobs=planner(rows)
    if sorted(i for j in jobs for i in j.commits)!=list(selected(rows)):
        raise ValueError('changed output mask')
    planned=time.perf_counter_ns();baseline=resize_bicubic(image);out=np.clip(baseline,0.,1.)
    fallback=time.perf_counter_ns();extract_ns=infer_ns=reconstruct_ns=0
    for j in jobs:
        if j.y%60 or j.x<0 or j.x+j.width>640 or any(i//10!=j.y//60 or i%10*64<j.x or (i%10+1)*64>j.x+j.width for i in j.commits):
            raise ValueError('invalid grouped region')
        t=time.perf_counter_ns()
        d=make_dispatch_descriptor(frame_id=0,frame_width=640,frame_height=360,core_x=j.x,core_y=j.y,core_width=j.width,core_height=60,halo=backend.halo,deadline_us=33333,priority=1.)
        if backend.boundary=='zero':p,x,y=extract_zero_halo_patch(image,d,halo=backend.halo)
        else:p=extract_halo_patch(image,d,halo=backend.halo);x=y=backend.halo
        t1=time.perf_counter_ns()
        boundary=('full_width_top' if j.y==0 else 'full_width_bottom' if j.y==300 else 'full_width_middle') if backend.boundary=='zero' and j.width==640 else None
        raw=backend.infer_boundary(boundary,(j.width,60),p) if boundary else backend.infer((j.width,60),p)
        t2=time.perf_counter_ns()
        residual=normalize_subpixel_residual(raw,input_width=p.shape[1],input_height=p.shape[0],quantization_scale=None,quantization_zero_point=None)
        for i in j.commits:
            r,c=divmod(i,10);sx=x+c*64-j.x
            out[r*120:(r+1)*120,c*128:(c+1)*128]=np.clip(baseline[r*120:(r+1)*120,c*128:(c+1)*128]+residual[y*2:y*2+120,sx*2:sx*2+128],0.,1.)
        t3=time.perf_counter_ns();extract_ns+=t1-t;infer_ns+=t2-t1;reconstruct_ns+=t3-t2
    end=time.perf_counter_ns()
    return out,{'service_ns':end-begin,'planning_ns':planned-begin,'fallback_ns':fallback-planned,
                'extraction_ns':extract_ns,'inference_ns':infer_ns,'reconstruction_ns':reconstruct_ns,'calls':len(jobs)}
