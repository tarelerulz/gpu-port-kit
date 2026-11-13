"""Repeat-run check at several clip lengths: every run must give bit-identical logits (catches ordering races in multi-head jobs)."""
import os, sys, numpy as np
os.environ['GPUPORT_ASM']='1'
KIT=os.path.expanduser('~/gpu-port-kit'); sys.path.insert(0,KIT); sys.path.insert(0,KIT+'/tests')
import nemo8_rebuild as nb
from gpuport import server
nr=nb.load_nemo_ref(); wt,L=nb.prepare(nr)
wpath=wt.save(os.path.join(nb.WORK_DIR,'nemo8_asm'))
pcm=np.fromfile(os.path.join(nb.SPEECH_APP,'clip10s.pcm'),'<i2')
srv=server.ExecServer(wpath,nb.SHADERS).start(); bad=0
for n in [int(a) for a in sys.argv[2:]]:
    plan,ar,R,T=nb.build(n,L,optimized=True); x=np.zeros(n,np.float32); x[:min(n,len(pcm))]=pcm[:n]/32768.0
    outs=[srv.run(plan,x,R['LOG'],T*(L['V']+1))[0] for _ in range(int(sys.argv[1]))]
    same=sum(np.array_equal(outs[0],o) for o in outs[1:]); nan=int(np.isnan(outs[0]).sum())
    print('T=%3d: identical %d/%d, NaN %d'%(T,same,len(outs)-1,nan)); bad+=(same!=len(outs)-1) or nan>0
print('DETERMINISM', 'PASS' if not bad else 'FAIL'); srv.proc.stdin.close(); srv.proc.wait(timeout=20)
