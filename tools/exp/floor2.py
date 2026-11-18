import os, sys, copy, time, numpy as np
os.environ['GPUPORT_ASM']='1'
KIT=os.path.expanduser('~/gpu-port-kit'); sys.path.insert(0,KIT); sys.path.insert(0,KIT+'/tests')
import nemo8_rebuild as nb
from gpuport import server
nr=nb.load_nemo_ref(); wt,L=nb.prepare(nr)
wpath=wt.save(os.path.join(nb.WORK_DIR,'nemo8_asm'))
pcm=np.fromfile(os.path.join(nb.SPEECH_APP,'clip10s.pcm'),'<i2')
srv=server.ExecServer(wpath,nb.SHADERS).start()
for n in (36800,160000):
    plan,ar,R,T=nb.build(n,L,optimized=True); x=pcm[:n].astype(np.float32)/32768.0
    def t(p,reps=9,xx=x,on=None):
        on = T*(L['V']+1) if on is None else on
        srv.run(p,xx,R['LOG'],on); ms=[]; wall=[]
        for _ in range(reps):
            t0=time.monotonic(); m=srv.run(p,xx,R['LOG'],on)[1]; wall.append((time.monotonic()-t0)*1e3); ms.append(m)
        return round(float(np.median(ms)),1), round(float(np.median(wall)),1)
    print('T',T,'full plan (server, wall):', t(plan))
    p1=copy.copy(plan); p1.ops=[plan.ops[-1]]
    print('  last op only:', t(p1), ' tiny input:', t(p1,xx=x[:16]), ' tiny input+tiny out:', t(p1,xx=x[:16],on=16))
    p3=copy.copy(plan); p3.ops=plan.ops[:1]*len(plan.ops)
    print('  same op count, first op repeated (%s):'%plan.ops[0].shader, t(p3))
srv.proc.stdin.close(); srv.proc.wait(timeout=20)
print(open('/sys/class/drm/card0/device/power/autosuspend_delay_ms').read() if os.path.exists('/sys/class/drm/card0/device/power/autosuspend_delay_ms') else 'no autosuspend file card0')
