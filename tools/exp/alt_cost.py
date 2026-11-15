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
Op=type(plan.ops[0])
isasm=lambda o:o.shader.startswith('qpu') and o.shader!='qpu_pack'
segs=sum(1 for i,o in enumerate(plan.ops) if not isasm(o) and (i==0 or isasm(plan.ops[i-1])))
print('ops',len(plan.ops),'asm',sum(map(isasm,plan.ops)),'vulkan segments',segs)
# A: trivial Vulkan op between two consecutive asm jobs (adds one alternation each)
ops=[];na=0
for i,o in enumerate(plan.ops):
    ops.append(o)
    if isasm(o) and i+1<len(plan.ops) and isasm(plan.ops[i+1]):
        ops.append(Op('add',128,24,1,[0,R['LOGP'],R['LOGP'],0])); na+=1
p2=copy.copy(plan); p2.ops=ops; ta=t(p2)
# B: the same number of trivial Vulkan ops placed right after an existing Vulkan op (no new alternation)
ops=[];nb_=0
for i,o in enumerate(plan.ops):
    ops.append(o)
    if not isasm(o) and nb_<na and i>20:
        ops.append(Op('add',128,24,1,[0,R['LOGP'],R['LOGP'],0])); nb_+=1
p3=copy.copy(plan); p3.ops=ops; tb=t(p3)
print('base',round(base),'| +%d alternations'%na,round(ta),'| +%d plain trivial ops'%nb_,round(tb),'| per alternation ms',round((ta-tb)/na,3),' per trivial op',round((tb-base)/nb_,3))
srv.proc.stdin.close(); srv.proc.wait(timeout=20)
