"""Detect a GPU reset during a request.

Finding (2026-11-08, tested): when a shader never finishes, the v3d kernel driver resets the GPU
(about 1 s later) and logs "Resetting GPU for hang" to the kernel log. Vulkan does NOT report it:
the fence and vkDeviceWaitIdle both return VK_SUCCESS, and the output is garbage. VK_EXT_device_fault
is not available on this driver. So the host compares the last reset line before and after a request.
Reading the kernel log needs no root on the test Pi (dmesg works for the user)."""
import ctypes
import re
import subprocess

RESET_TEXT = 'Resetting GPU for hang'


class GpuHang(RuntimeError):
    """A GPU reset happened during the request: its output must not be used."""


_KLOG_BUF = ctypes.create_string_buffer(4 << 20)


def _kernel_log_text():
    """The whole kernel ring buffer via klogctl(SYSLOG_ACTION_READ_ALL), ~2 ms (2026-11-09).
    Falls back to dmesg (a subprocess, ~25 ms) if klogctl is not allowed."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        n = libc.klogctl(3, _KLOG_BUF, len(_KLOG_BUF))
        if n >= 0:
            return _KLOG_BUF.raw[:n].decode(errors='replace')
    except Exception:
        pass
    return subprocess.run(['dmesg'], capture_output=True, text=True, timeout=10).stdout


def last_reset_marker():
    """The last reset line in the kernel log ('' if none, or None if the log cannot be read)."""
    try:
        out = _kernel_log_text()
    except Exception:
        return None                      # unknown: caller treats as "cannot tell"
    marker = ''
    for line in out.splitlines():
        if RESET_TEXT in line:
            marker = re.sub(r'^<\d+>', '', line)     # klogctl keeps the priority prefix, dmesg does not
    return marker


def check_no_reset(before):
    """Raise GpuHang if the kernel logged a new reset since `before` (from last_reset_marker())."""
    after = last_reset_marker()
    if before is None or after is None:
        raise RuntimeError('cannot read the kernel log to check for a GPU reset (dmesg)')
    if after != before:
        raise GpuHang('GPU reset during this request (kernel: ' + after.strip()[-80:] + '); output discarded')


def gpu_reason_since(marker):
    """The last v3d kernel messages logged after `marker` (from last_reset_marker()), one line.
    Used in the error message when the server exits with a GPU error."""
    try:
        lines = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=10).stdout.splitlines()
    except Exception:
        return '(kernel log not readable)'
    if marker and marker in lines:
        idx = len(lines) - 1 - lines[::-1].index(marker)
        lines = lines[idx + 1:]
    msgs = [l.split('] ', 1)[-1].strip() for l in lines if 'v3d' in l]
    return ' | '.join(msgs[-3:]) if msgs else '(no v3d messages in the kernel log)'
