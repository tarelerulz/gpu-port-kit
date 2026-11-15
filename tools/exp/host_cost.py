import os, sys, time, numpy as np
os.environ['GPUPORT_ASM']='1'
KIT=os.path.expanduser('~/gpu-port-kit'); sys.path.insert(0,KIT); sys.path.insert(0,KIT+'/tests')
import nemo8_rebuild as nb
from gpuport import nemo8, gpu_health, config
pcm=np.fromfile(os.path.join(nb.SPEECH_APP,'clip10s.pcm'),'<i2')
e=nemo8.Nemo8Encoder(); e.start(); L=e.L
def med(f,n=7):
    v=[]
    for _ in range(n):
        t0=time.monotonic(); r=f(); v.append((time.monotonic()-t0)*1e3)
    return round(float(np.median(v)),1)
print('transcribe wall ms', med(lambda: e.transcribe(pcm)), ' server ms', e.last_ms)
print('build plan ms', med(lambda: nemo8.build(len(pcm), L)))
plan,ar,R,T=nemo8.build(len(pcm),L)
base=os.path.join(config.WORK_DIR,'hc_test')
print('plan.write ms', med(lambda: plan.write(base+'.plan')), ' size', os.path.getsize(base+'.plan'))
x=pcm.astype(np.float32)/32768.0
print('arena tofile ms', med(lambda: x.astype('<f4').tofile(base+'.arena')))
np.zeros(T*1025,np.float32).tofile(base+'.out'); print('out fromfile ms', med(lambda: np.fromfile(base+'.out','<f4')))
print('last_reset_marker ms', med(gpu_health.last_reset_marker)); m=gpu_health.last_reset_marker()
print('check_no_reset ms', med(lambda: gpu_health.check_no_reset(m)))
print('pcm->float ms', med(lambda: pcm.astype(np.float32)/32768.0))
import inspect; print(inspect.getsource(nemo8.Nemo8Encoder.transcribe))
print('work dir', config.WORK_DIR); e.close()
