"""A/B: text vs binary plan, spinner on/off. Same server binary. Wall + server ms + logits identical check."""
import os, sys, time, numpy as np
os.environ['GPUPORT_ASM']='1'
KIT=os.path.expanduser('~/gpu-port-kit'); sys.path.insert(0,KIT); sys.path.insert(0,KIT+'/tests')
import nemo8_rebuild as nb
from gpuport import server, nemo8
nr=nb.load_nemo_ref(); wt,L=nb.prepare(nr)
wpath=wt.save(os.path.join(nb.WORK_DIR,'nemo8_asm'))
pcm=np.fromfile(os.path.join(nb.SPEECH_APP,'clip10s.pcm'),'<i2')
def mk(bplan, spin):
    os.environ['GPUPORT_BPLAN']=bplan; os.environ['GPUPORT_SPIN']=spin
    return server.ExecServer(wpath,nb.SHADERS).start()
cfg=[('text nospin','0','0'),('bin nospin','1','0'),('text spin','0','1'),('bin spin','1','1')]
srvs={c[0]:mk(c[1],c[2]) for c in cfg}
ref={}
for rnd in range(2):
  for n in (36800,80000,160000):
    x=pcm[:n].astype(np.float32)/32768.0
    for name in srvs:
        srv=srvs[name]; ms=[]; wall=[]
        for i in range(8):
            time.sleep(0.3)                                   # idle gap like dictation: CPU clock drops
            t0=time.monotonic(); plan,ar,R,T=nemo8.build(n,L); out,m=srv.run(plan,x,R['LOG'],T*(L['V']+1)); wall.append((time.monotonic()-t0)*1e3); ms.append(m)
        same = np.array_equal(ref.setdefault(n,out), out)
        print('T=%3d %-12s server %6.1f  wall(build+run) %6.1f  identical %s'%(T,name,np.median(ms),np.median(wall),same),flush=True)
for s in srvs.values(): s.stop()
