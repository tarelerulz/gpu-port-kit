"""Timing-only: time saved when all ops of one type are removed from the plan (output garbage). usage: skip_by_op.py [samples]"""
import os, sys, copy, numpy as np
os.environ['GPUPORT_ASM']='1'
KIT=os.path.expanduser('~/gpu-port-kit'); sys.path.insert(0,KIT); sys.path.insert(0,KIT+'/tests')
import nemo8_rebuild as nb
from gpuport import server
nr=nb.load_nemo_ref(); wt,L=nb.prepare(nr)
wpath=wt.save(os.path.join(nb.WORK_DIR,'nemo8_asm'))
pcm=np.fromfile(os.path.join(nb.SPEECH_APP,'clip10s.pcm'),'<i2')
n=int(sys.argv[1]) if len(sys.argv)>1 else len(pcm)
plan,ar,R,T=nb.build(n,L,optimized=True)
x=np.zeros(n,np.float32); x[:min(n,len(pcm))]=pcm[:n]/32768.0
srv=server.ExecServer(wpath,nb.SHADERS).start()
def t(p,reps=5):
    srv.run(p,x,R['LOG'],T*(L['V']+1))
    return float(np.median([srv.run(p,x,R['LOG'],T*(L['V']+1))[1] for _ in range(reps)]))
base=t(plan,9); names=sorted(set(o.shader for o in plan.ops)); d={}
import collections; cnt=collections.Counter(o.shader for o in plan.ops)
for nm in names:
    p2=copy.copy(plan); p2.ops=[o for o in plan.ops if o.shader!=nm]
    if not p2.ops or p2.ops[-1].shader.startswith('qpu'): p2.ops=p2.ops+[plan.ops[-1]]
    d[nm]=base-t(p2)
print('T',T,'base',round(base),'ops',len(plan.ops))
for nm in sorted(d,key=lambda k:-d[k]): print('%-16s x%-4d %6.1f ms'%(nm,cnt[nm],d[nm]))
print('sum',round(sum(d.values())))
srv.proc.stdin.close(); srv.proc.wait(timeout=20)
