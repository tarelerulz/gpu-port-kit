import os, sys, copy, time, numpy as np
os.environ['GPUPORT_ASM']='1'
KIT=os.path.expanduser('~/gpu-port-kit'); sys.path.insert(0,KIT); sys.path.insert(0,KIT+'/tests')
import nemo8_rebuild as nb
from gpuport import server
nr=nb.load_nemo_ref(); wt,L=nb.prepare(nr)
wpath=wt.save(os.path.join(nb.WORK_DIR,'nemo8_asm'))
pcm=np.fromfile(os.path.join(nb.SPEECH_APP,'clip10s.pcm'),'<i2'); n=len(pcm)
plan,ar,R,T=nb.build(n,L,optimized=True)
x=pcm.astype(np.float32)/32768.0
srv=server.ExecServer(wpath,nb.SHADERS).start()
def t(p,reps=7):
    srv.run(p,x,R['LOG'],T*(L['V']+1)); ms=[]; wall=[]
    for _ in range(reps):
        t0=time.monotonic(); m=srv.run(p,x,R['LOG'],T*(L['V']+1))[1]; wall.append((time.monotonic()-t0)*1e3); ms.append(m)
    return round(float(np.median(ms)),1), round(float(np.median(wall)),1)
print('full plan (server ms, wall ms):', t(plan))
p1=copy.copy(plan); p1.ops=[plan.ops[-1]]; print('only last op:', t(p1))
vk=[o for o in plan.ops if not o.shader.startswith('qpu') or o.shader=='qpu_pack']
p2=copy.copy(plan); p2.ops=vk; print('all Vulkan ops, no asm (%d ops):'%len(vk), t(p2))
tiny=copy.copy(plan); Op=type(plan.ops[0]); tiny.ops=[Op('add',128,24,1,[0,R['LOGP'],R['LOGP'],0]) for _ in range(len(vk))]; print('%d trivial Vulkan ops:'%len(vk), t(tiny))
srv.proc.stdin.close(); srv.proc.wait(timeout=20)
