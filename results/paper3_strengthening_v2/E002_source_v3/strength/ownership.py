"""Authenticated dual-node supplement to accepted RK3576 telemetry.

Helper stdout/errors are returned for journaling before pass/fail decisions.
This does not edit or upgrade the historical single-node source records.
"""
import hashlib
import json
from pathlib import Path
import stat
import subprocess

HELPER=Path('/usr/local/libexec/paper3-rk3576-npu-owners-v2')
SHA='c86a1ae79a72a6955b8d0dc15c75e36e2fcbe2d332b981f4d7dde69022ecb4a7'


def proof():
    info=HELPER.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or info.st_gid!=0 or stat.S_IMODE(info.st_mode)!=0o555 or hashlib.sha256(HELPER.read_bytes()).hexdigest()!=SHA:raise ValueError('dual-node helper identity drift')
    try:
        result=subprocess.run(['/usr/bin/sudo','-n',str(HELPER)],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30,check=False,cwd='/',env={'PATH':'/usr/sbin:/usr/bin:/sbin:/bin','LANG':'C','LC_ALL':'C'})
        stdout,stderr,code=result.stdout,result.stderr,result.returncode
    except subprocess.TimeoutExpired as error:stdout,stderr,code=error.stdout or b'',error.stderr or b'',None
    try:
        value=json.loads(stdout)
        if not isinstance(value,dict):raise ValueError('non-object helper JSON')
        parse_error=None
    except (ValueError,UnicodeError) as error:value={};parse_error=repr(error)
    return value,{'returncode':code,'stdout':stdout.decode(errors='replace'),'stderr':stderr.decode(errors='replace'),'parse_error':parse_error,'helper_sha256':SHA,'stdout_sha256':hashlib.sha256(stdout).hexdigest()}


def attach(snapshot,value,capture,owned_pids):
    # Historical proof remains visible; supplement has distinct scope.
    snapshot['old_render_node_proof']=snapshot['npu_ownership_proof']
    sweeps=value.get('sweeps',[]);owners=sweeps[0] if isinstance(sweeps,list) and sweeps and isinstance(sweeps[0],list) else []
    digest=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    valid=(capture['returncode']==0 and not capture['stderr'] and capture['parse_error'] is None
        and value.get('schema')==1 and value.get('scope')=='RK3576_DUAL_NODE_NPU_OWNERS_ONLY'
        and value.get('complete') is True and value.get('errors')==[]
        and value.get('covered_devices')==[[226,1],[226,129]]
        and len(sweeps)==2 and sweeps[0]==sweeps[1] and digest(owners)==value.get('owner_vector_sha256')
        and all(isinstance(row,dict) and type(row.get('pid')) is int and row.get('device') in ([226,1],[226,129]) for row in owners))
    snapshot['dual_node_capture']=capture
    snapshot['npu_ownership_complete']=valid
    snapshot['npu_ownership_proof']={'owners':owners,'coverage':value,'helper_sha256':SHA}
    snapshot['invariants']['other_npu_owners']=[row for row in owners if not isinstance(row,dict) or row.get('pid') not in owned_pids]
    return snapshot
