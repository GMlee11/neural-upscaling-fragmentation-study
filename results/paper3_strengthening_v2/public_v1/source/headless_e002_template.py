"""Fixed E002 headless sessions with service restoration between sessions.

Builder replaces SOURCE_ID and SOURCE_SHA tokens after source freeze. Root
controls only the two named services; inference runs as radxa. No arguments.
"""
import sys
sys.dont_write_bytecode=True
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/home/radxa/experiments/paper3-strengthening-v2')
SOURCE=ROOT/'SOURCE_ID_TOKEN'
SHA='SOURCE_SHA_TOKEN'
SERVICES=('gdm.service','systemd-logind.service')


def state(service):
    text=subprocess.check_output(['/usr/bin/systemctl','show',service,'-p','ActiveState','-p','UnitFileState','-p','MainPID','-p','NFileDescriptorStore'],text=True,timeout=30)
    return dict(line.split('=',1) for line in text.splitlines())


def owners():
    value=json.loads(subprocess.check_output(['/usr/local/libexec/paper3-rk3576-npu-owners-v2'],text=True,timeout=30))
    if value.get('complete') is not True:raise ValueError('incomplete ownership proof')
    return value


def main():
    if len(sys.argv)!=1 or os.geteuid()!=0:raise ValueError('fixed no-argument root launcher')
    raw=(SOURCE/'source_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA:raise ValueError('source manifest mismatch')
    for rel,pin in json.loads(raw)['files'].items():
        if hashlib.sha256((SOURCE/rel).read_bytes()).hexdigest()!=pin:raise ValueError('frozen source changed')
    path=ROOT/(SOURCE.name+'_headless.jsonl')
    with path.open('x',buffering=1) as log:
        def emit(event):
            log.write(json.dumps(dict(event,unix_time=time.time()),sort_keys=True)+'\n')
            log.flush();os.fsync(log.fileno())
        def interrupted(signum,frame):raise SystemExit('launcher interrupted '+str(signum))
        signal.signal(signal.SIGTERM,interrupted)
        try:
            for session in (0,1):
                prior={service:state(service) for service in SERVICES};stopped=[];child=None
                emit({'state':'session_initial','session':session,'services':prior,'owners':owners()})
                try:
                    if any(item['ActiveState'] not in ('active','inactive') for item in prior.values()):
                        raise ValueError('unstable initial service state')
                    if prior['gdm.service']['ActiveState']=='active':
                        stopped.append('gdm.service')
                        subprocess.run(['/usr/bin/systemctl','stop','gdm.service'],check=True,timeout=60)
                    proof=owners();emit({'state':'after_display_stop','session':session,'owners':proof})
                    if proof['sweeps'][0] and prior['systemd-logind.service']['ActiveState']=='active':
                        stopped.append('systemd-logind.service')
                        subprocess.run(['/usr/bin/systemctl','stop','systemd-logind.service'],check=True,timeout=60)
                    proof=owners();emit({'state':'headless','session':session,'owners':proof,'services':{s:state(s) for s in SERVICES}})
                    if proof['sweeps'][0]:raise ValueError('foreign handles remain; no samples')
                    cells=('width24','fp16') if session==0 else ('fp16','width24')
                    for cell in cells:
                        run_id='E002_'+cell+'_s'+str(session)+'_'+SOURCE.name.removeprefix('E002_source_')
                        with (ROOT/(run_id+'_console.log')).open('xb') as output:
                            args=['/usr/sbin/runuser','-u','radxa','--','/usr/bin/python3','-B',str(SOURCE/'run_e002.py'),
                                  '--cell',cell,'--session',str(session),'--run-id',run_id,'--source-manifest-sha',SHA]
                            child=subprocess.Popen(args,cwd=SOURCE,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                            emit({'state':'running_cell','session':session,'cell':cell,'run_id':run_id,'child_pid':child.pid})
                            code=child.wait(timeout=3600);child=None
                        emit({'state':'cell_exit','session':session,'cell':cell,'returncode':code})
                        if code!=0:raise RuntimeError('frozen cell failed; no next cell')
                finally:
                    if child is not None and child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                    failed=[]
                    for service in reversed(stopped):
                        try:
                            code=subprocess.run(['/usr/bin/systemctl','start',service],check=False,timeout=60).returncode
                            emit({'state':'restoration','session':session,'service':service,'returncode':code,'after':state(service)})
                            if code:failed.append(service)
                        except (OSError,subprocess.SubprocessError) as error:
                            failed.append(service);emit({'state':'restoration_failure','session':session,'service':service,'error':repr(error)})
                    after={service:state(service) for service in SERVICES}
                    emit({'state':'restoration_complete','session':session,'services':after})
                    if failed or any(after[s]['ActiveState']!=prior[s]['ActiveState'] or after[s]['UnitFileState']!=prior[s]['UnitFileState'] for s in SERVICES):
                        raise RuntimeError('service restoration mismatch; no further session')
                if session==0:
                    emit({'state':'fixed_idle','seconds':300});time.sleep(300)
            emit({'state':'complete','sessions':2,'cells':4})
        except BaseException as error:emit({'state':'failed','error':repr(error)});raise


if __name__=='__main__':main()
