"""A/B GPUPORT_ASM_TAIL1 (1-vector tail kernel) on one server, alternating plans, median of 15. usage: ab_tail.py n..."""
import os, sys, numpy as np
os.environ['GPUPORT_ASM']='1'
KIT=os.path.expanduser('~/gpu-port-kit'); sys.path.insert(0,KIT); sys.path.insert(0,KIT+'/tests')
import nemo8_rebuild as nb
from gpuport import server, nemo8
nr=nb.load_nemo_ref(); wt,L=nb.prepare(nr)
wpath=wt.save(os.path.join(nb.WORK_DIR,'nemo8_asm'))
pcm=np.fromfile(os.path.join(nb.SPEECH_APP,'clip10s.pcm'),'<i2')
srv=server.ExecServer(wpath,nb.SHADERS).start()
for n in [int(a) for a in sys.argv[1:]]:
    x=np.resize(pcm,n).astype(np.float32)/32768.0; P={}
    for v in ('0','1'):
        os.environ['GPUPORT_ASM_TAIL1']=v; P[v]=nemo8.build(n,L)
    T=P['0'][3]; R=P['0'][2]; ms={'0':[],'1':[]}; outs={}
    for i in range(16):
        for v in ('0','1'):
            o,m=srv.run(P[v][0],x,R['LOG'],T*(L['V']+1)); outs[v]=o
            if i: ms[v].append(m)
    d=float(np.abs(outs['0']-outs['1']).max())
    n1=sum(o.shader.startswith('qpu1_') for o in P['1'][0].ops)
    print('%5.2f s T=%3d  old tail %6.1f ms  new tail %6.1f ms  (%+5.1f)  qpu1 jobs %d  max logit diff %.3g  same argmax %s'%(n/16000,T,np.median(ms['0']),np.median(ms['1']),np.median(ms['1'])-np.median(ms['0']),n1,d,bool((outs['0'].reshape(T,-1).argmax(1)==outs['1'].reshape(T,-1).argmax(1)).all())),flush=True)
srv.stop()
