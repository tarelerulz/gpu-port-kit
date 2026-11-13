"""Minimal local stand-in for the PyPI ioctl_opt package (only IO/IOR/IOW/IOWR, ARM/aarch64 Linux ioctl number layout).
Written so the py-videocore6 driver imports without pip. PYTHONPATH=qpu/shim"""
import ctypes
_NRBITS, _TYPEBITS, _SIZEBITS = 8, 8, 14
_NRSHIFT = 0
_TYPESHIFT = _NRSHIFT + _NRBITS
_SIZESHIFT = _TYPESHIFT + _TYPEBITS
_DIRSHIFT = _SIZESHIFT + _SIZEBITS
_NONE, _WRITE, _READ = 0, 1, 2
def IOC(d, t, nr, size):
    if isinstance(t, str): t = ord(t)
    return (d << _DIRSHIFT) | (t << _TYPESHIFT) | (nr << _NRSHIFT) | (size << _SIZESHIFT)
def _sz(s): return s if isinstance(s, int) else ctypes.sizeof(s)
def IO(t, nr): return IOC(_NONE, t, nr, 0)
def IOR(t, nr, s): return IOC(_READ, t, nr, _sz(s))
def IOW(t, nr, s): return IOC(_WRITE, t, nr, _sz(s))
def IOWR(t, nr, s): return IOC(_READ | _WRITE, t, nr, _sz(s))
