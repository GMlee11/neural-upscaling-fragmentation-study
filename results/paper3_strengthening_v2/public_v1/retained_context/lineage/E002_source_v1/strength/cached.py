"""Conventional table compilation of unchanged finite-row planners.

No new scheduling algorithm. Preparation is charged separately; lookup time
still belongs inside complete request service. Valid only for our supplied
costs, which depend on row class/width/commit count, not the middle-row index.
"""
from dataclasses import dataclass
import hashlib
import json
import time
from types import MappingProxyType

from .core import affine_cost, lookup_cost, partition, per_row, row_class, verify_jobs,grouped_rows
from src.experiments.paper3_dispatch_successor_v1.core import Job,validate_mask,plan

POLICIES=('exact','row_spans','grouped','per_row','affine_partition','lookup_partition')
MODES=('whole','selected')
CANDIDATES=tuple((policy,mode) for policy in POLICIES for mode in MODES)


@dataclass(frozen=True)
class CompiledRowPlanner:
    table: object
    policy: str
    cost_sha256: str
    table_sha256: str
    serialized_table_bytes: int
    preparation_ns: int

    def __call__(self, rows):
        rows=validate_mask(rows)
        jobs=[]
        for r,bits in enumerate(rows):
            for x,width,columns in self.table[row_class(r*60),int(bits,2)]:
                jobs.append(Job(x,r*60,width,tuple(r*10+c for c in columns)))
        return verify_jobs(rows,tuple(jobs))


def compile_rows(profile,policy):
    start=time.perf_counter_ns()
    if policy=='affine_partition':
        cost=affine_cost(profile['coefficients']);solver=partition
        relevant={'coefficients':profile['coefficients']}
    elif policy in ('lookup_partition','per_row'):
        cost=lookup_cost(profile['costs']);solver=partition if policy=='lookup_partition' else per_row
        relevant={'costs':profile['costs']}
    else:
        raise ValueError('only unchanged calibrated planners may be table compiled')
    cost_raw=json.dumps(relevant,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    table={}
    for row in (0,1,5):
        table[row_class(row*60),0]=()
        for bits in range(1,1024):
            mask=tuple(format(bits,'010b') if r==row else '0000000000' for r in range(6))
            jobs=solver(mask,cost)
            table[row_class(row*60),bits]=tuple((j.x,j.width,tuple(i%10 for i in j.commits)) for j in jobs)
    canonical=[{'row_class':key[0],'mask':key[1],'jobs':value} for key,value in sorted(table.items())]
    raw=json.dumps(canonical,separators=(',',':'),allow_nan=False).encode()
    return CompiledRowPlanner(MappingProxyType(table),policy,hashlib.sha256(cost_raw).hexdigest(),
                              hashlib.sha256(raw).hexdigest(),len(raw),time.perf_counter_ns()-start)


@dataclass(frozen=True)
class CompiledChoice:
    planners: object
    scores: object
    preparation_ns: int
    score_sha256: str
    serialized_score_bytes: int

    def planner(self,policy,mode):
        if (policy,mode) not in CANDIDATES:raise ValueError('unknown frozen candidate')
        if policy in ('exact','row_spans'):return lambda rows:plan(rows,policy)
        if policy=='grouped':return grouped_rows
        return self.planners[policy,mode]

    def decide(self,rows):
        rows=validate_mask(rows)
        keys=tuple((row_class(r*60),int(bits,2)) for r,bits in enumerate(rows))
        costs=[sum(self.scores[p,m,kind,bits] for kind,bits in keys) for p,m in CANDIDATES]
        # Exact frozen candidate order resolves equal floating-point costs.
        winner=min(range(len(CANDIDATES)),key=lambda i:(costs[i],i))
        policy,mode=CANDIDATES[winner]
        return policy,mode,costs[winner]


def compile_choice(profiles):
    start=time.perf_counter_ns()
    planners={(p,m):compile_rows(profiles[m],p) for p in POLICIES[3:] for m in MODES}
    scores={}
    for mode in MODES:
        cost=lookup_cost(profiles[mode]['costs'])
        for policy in POLICIES:
            for row in (0,1,5):
                kind=row_class(row*60);scores[policy,mode,kind,0]=0.
                for bits in range(1,1024):
                    mask=tuple(format(bits,'010b') if r==row else '0000000000' for r in range(6))
                    if policy in ('exact','row_spans'):jobs=plan(mask,policy)
                    elif policy=='grouped':jobs=grouped_rows(mask)
                    else:jobs=planners[policy,mode](mask)
                    scores[policy,mode,kind,bits]=sum(cost(job) for job in jobs)
    raw=json.dumps([(*key,value) for key,value in sorted(scores.items())],separators=(',',':'),allow_nan=False).encode()
    return CompiledChoice(MappingProxyType(planners),MappingProxyType(scores),time.perf_counter_ns()-start,
                          hashlib.sha256(raw).hexdigest(),len(raw))
