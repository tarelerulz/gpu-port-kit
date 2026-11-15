import os, sys, copy, numpy as np
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
def t(p,reps=9):
    srv.run(p,x,R['LOG'],T*(L['V']+1))
    return float(np.median([srv.run(p,x,R['LOG'],T*(L['V']+1))[1] for _ in range(reps)]))
base=t(plan)
a=L['asm'][0]['ff1_1']
TP=256
res={}
for per in (1,3):
    p2=copy.copy(plan); ops=[]
    for o in plan.ops:
        ops.append(o)
        if o.shader=='qpu_mmsk':
            for _ in range(per):
                d=type(o)('qpu_mm',256,1,1,[2,1,R['X'],TP,R['LOGP'],TP,a['wo'],32,16])
                ops.append(d)
    p2.ops=ops; res[per]=t(p2)
    print('extra jobs',64*per,'time',round(res[per]),'base',round(base),'per job ms',round((res[per]-base)/(64*per),3))
srv.proc.stdin.close(); srv.proc.wait(timeout=20)
