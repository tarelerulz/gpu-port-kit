"""Check 1: Vulkan arena (dma-buf) shared with a hand-written QPU job, ordering by the kernel's implicit sync only. 2026-11-12.
Round: Vulkan chain writes A (submitted, not waited) -> python submits the assembly matmul CSD job on the imported BO (not waited)
-> Vulkan job adds 1.0 to O (submitted) -> Vulkan waits and dumps O -> python compares with numpy.
usage: PYTHONPATH=<py-videocore6>/src python3 check1.py [rounds] [nadd]"""
import fcntl, os, socket, struct, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
KIT = '/home/tarel/gpu-port-kit'
sys.path.insert(0, KIT)
import mm_asm as M
from videocore6.driver import Driver
import _videocore6.driver as D
from _videocore6.driver import Dispatcher
_orig_drm = D.DRM_V3D
D.DRM_V3D = lambda *a, **k: _orig_drm(os.environ.get('V3D_NODE', '/dev/dri/renderD128'))   # render node: no DRM master / auth needed while Vulkan holds card0
from gpuport.config import VK_ENV, merged_shader_dir

WAIT1 = os.environ.get('WAIT1', '0'); WAITCSD = os.environ.get('WAITCSD') == '1'
EXPLICIT = os.environ.get('EXPLICIT') == '1'
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
nadd = int(sys.argv[2]) if len(sys.argv) > 2 else 300
K, NT = 704, 11
N = 16 * NT
S = (K + 4) * 256

HERE = os.path.dirname(os.path.abspath(__file__))
subprocess.run(['gcc', '-O2', '-o', HERE + '/vkshare', HERE + '/vkshare.c', '-lvulkan', '-L' + os.path.expanduser('~/opt/pios-vulkan/lib')], check=True)
a_sock, c_sock = socket.socketpair()
outp = '/tmp/claude-1000/-home-tarel/4c2a1058-be79-4465-9615-e64e4bca54c2/scratchpad/vkshare_O.bin'
env = dict(os.environ, **VK_ENV)
proc = subprocess.Popen([HERE + '/vkshare', str(c_sock.fileno()), outp, str(K), str(N), str(nadd), str(rounds), WAIT1, '1' if EXPLICIT else '0'],
                        pass_fds=[c_sock.fileno()], env=env, cwd=merged_shader_dir())
c_sock.close()
msg, fds, _, _ = socket.recv_fds(a_sock, 16, 1)
if msg != b'R' or not fds:
    print('FAIL: Vulkan side could not export the arena (message %r)' % msg); proc.wait(); sys.exit(1)
dmafd = fds[0]
print('got dma-buf fd', dmafd)

rng = np.random.default_rng(3)
Wref = (rng.standard_normal((K, N)) * 0.05).astype(np.float32)
Aexp = np.zeros((K, 251), np.float32)
kk = np.arange(K)[:, None]; mm = np.arange(251)[None, :]
A0 = (((kk * 31 + mm * 7) % 17) * 0.125).astype(np.float32)
Aexp = A0 + float(nadd)                                       # every element got += 1.0 nadd times
want = (Aexp.T @ Wref).T + 1.0                                # then the Vulkan job adds 1.0


def _ioctl(fd, nr, fmt, *vals):
    size = struct.calcsize(fmt)
    req = (3 << 30) | (size << 16) | (ord('d') << 8) | nr
    buf = bytearray(struct.pack(fmt, *vals))
    fcntl.ioctl(fd, req, buf, True)
    return struct.unpack(fmt, bytes(buf))


def syncobj_create(fd):
    return _ioctl(fd, 0xBF, 'II', 0, 0)[0]


def syncobj_import_syncfile(fd, handle, syncfd):
    _ioctl(fd, 0xC2, 'IIiIQ', handle, 1, syncfd, 0, 0)             # DRM_SYNCOBJ_FD_TO_HANDLE, IMPORT_SYNC_FILE


def syncobj_export_syncfile(fd, handle):
    return _ioctl(fd, 0xC1, 'IIiIQ', handle, 1, -1, 0, 0)[2]       # DRM_SYNCOBJ_HANDLE_TO_FD, EXPORT_SYNC_FILE


def csd_submit(drm, handles, code, uniforms, thread, in_sync, out_sync):
    wg_size = 16
    cfg = (1 << 16, 1 << 16, 1 << 16, ((16 * wg_size + 15) // 16 - 1) << 12 | (16 << 8) | (wg_size & 0xFF), thread - 1,
           int(code.addresses()[0]), int(uniforms))
    drm.v3d_submit_csd(cfg=cfg, coef=(0, 0, 0, 0), bo_handles=handles.ctypes.data, bo_handle_count=len(handles), in_sync=in_sync, out_sync=out_sync)


with Driver() as drv:
    drm = drv.drm
    # import the dma-buf into the library's DRM fd and ask for its GPU address
    req = bytearray(struct.pack('IIi', 0, 0, dmafd))
    DRM_IOCTL_PRIME_FD_TO_HANDLE = (3 << 30) | (12 << 16) | (ord('d') << 8) | 0x2e
    fcntl.ioctl(drm.fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, req, True)
    handle = struct.unpack('IIi', bytes(req))[0]
    req = bytearray(struct.pack('II', handle, 0))
    DRM_IOCTL_V3D_GET_BO_OFFSET = (3 << 30) | (8 << 16) | (ord('d') << 8) | (0x40 + 5)
    fcntl.ioctl(drm.fd, DRM_IOCTL_V3D_GET_BO_OFFSET, req, True)
    base = struct.unpack('II', bytes(req))[1]
    print('imported handle', handle, 'GPU address 0x%x' % base)

    code = drv.program(M.kernel)
    Wt = drv.alloc((NT + 1, K, 16), dtype=np.float32)
    Wt[:] = 0.0
    for i in range(NT):
        Wt[i] = Wref[:, 16 * i:16 * i + 16]
    MP = 256
    u = [base, MP * 4, MP * 4 + 64, 2 * MP * 4, 64, base + 2 * S * 4, MP * 4, MP * 4 - 64, 16 * MP * 4,
         Wt.addresses()[0, 0, 0], K * 16 * 4, K // 2, NT]
    unif = drv.alloc(len(u) + 2, dtype=np.uint32)
    unif[:len(u)] = np.array(u, dtype=np.uint64).astype(np.uint32)
    handles = np.append(drv.bo_handles, np.uint32(handle)).astype(np.uint32)

    bad = 0
    for r in range(rounds):
        a_sock.send(b'1')
        if EXPLICIT:
            msg, fds, _, _ = socket.recv_fds(a_sock, 1, 1)
            assert msg == b'S' and fds
            t0 = time.monotonic()
            h_in = syncobj_create(drm.fd); syncobj_import_syncfile(drm.fd, h_in, fds[0]); os.close(fds[0])
            h_out = syncobj_create(drm.fd)
            csd_submit(drm, handles, code, unif.addresses()[0], 8, h_in, h_out)
            fd2 = syncobj_export_syncfile(drm.fd, h_out)
            t_sub = time.monotonic() - t0
            socket.send_fds(a_sock, [b'2'], [fd2]); os.close(fd2)
        else:
            assert a_sock.recv(1) == b'S'                         # Vulkan chain is submitted, probably still running
            time.sleep(float(os.environ.get('DELAY_MS', '0')) / 1e3)   # let the (threaded) Vulkan submit reach the kernel first
            t0 = time.monotonic()
            with Dispatcher(drm, handles, timeout_sec=20) as csd:
                csd.dispatch(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0], thread=8)
                t_sub = time.monotonic() - t0
                if not WAITCSD:
                    a_sock.send(b'2')
            if WAITCSD:
                a_sock.send(b'2')
        assert a_sock.recv(1) == b'D'
        got = np.fromfile(outp, '<f4').reshape(N, 256)[:, :251]
        err = float(np.abs(got - want).max())
        ok = err < 1e-2
        bad += (not ok)
        print(f'round {r}: csd submit {t_sub * 1e3:.2f} ms, max err vs numpy {err:.3g} ' + ('OK' if ok else 'MISMATCH (ordering or sharing broken)'))
print('RESULT:', 'ordering OK in all rounds' if bad == 0 else f'{bad} of {rounds} rounds wrong')
proc.wait()
