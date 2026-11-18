"""End-to-end Nemo8Encoder.transcribe wall time with idle gaps (as in dictation). OLD=1: text plan, no spin, no wake."""
import os, sys, time, numpy as np
os.environ['GPUPORT_ASM']='1'
if os.environ.get('OLD')=='1': os.environ.update(GPUPORT_BPLAN='0',GPUPORT_SPIN='0',GPUPORT_WAKE='0')
KIT=os.path.expanduser('~/gpu-port-kit'); sys.path.insert(0,KIT); sys.path.insert(0,KIT+'/tests')
import nemo8_rebuild as nb
from gpuport import nemo8
pcm=np.fromfile(os.path.join(nb.SPEECH_APP,'clip10s.pcm'),'<i2')
e=nemo8.Nemo8Encoder(); e.start()
for n in (24000,36800,45000,80000,124321,160000):
    w=[]; s=[]; txt=None
    for i in range(9):
        time.sleep(0.4)
        t0=time.monotonic(); txt=e.transcribe(pcm[:n]); w.append((time.monotonic()-t0)*1e3); s.append(e.last_ms)
    print('%5.2f s  wall %6.1f  server %6.1f  host %5.1f  | %s'%(n/16000,np.median(w),np.median(s),np.median(w)-np.median(s),txt[:60]),flush=True)
e.close()
