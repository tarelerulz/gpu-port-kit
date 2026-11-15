"""Arena: one float buffer shared by all ops of a plan. Regions are placed with alignment and a gap."""
from .config import ALIGN, REGION_GAP, EXEC_ARENA_FLOATS


def roundup(a, b):
    return (a + b - 1) // b * b


class Arena:
    def __init__(self, limit=EXEC_ARENA_FLOATS, gap=REGION_GAP, align=ALIGN):
        self.limit, self.gap, self.align = limit, gap, align
        self.cur = 0
        self.regions = {}   # name -> (offset, nfloats)

    def alloc(self, name, nfloats):
        if name in self.regions:
            raise ValueError(f'arena region {name!r} allocated twice')
        off = roundup(self.cur, self.align)
        self.regions[name] = (off, int(nfloats))
        self.cur = off + int(nfloats) + self.gap
        if self.cur > self.limit:
            raise MemoryError(f'arena needs {self.cur} floats, limit {self.limit} '
                              f'(exec batch arena is {EXEC_ARENA_FLOATS}); reduce regions or use exec_server2')
        return off

    def off(self, name):
        return self.regions[name][0]

    def size(self, name):
        return self.regions[name][1]

    def __getitem__(self, name):
        return self.off(name)

    def used(self):
        return self.cur

    def where(self, off):
        """Name the place of a float offset: 'X+12' inside region X, 'gap after X+5', or 'beyond arena'."""
        last = None
        for name, (o, n) in sorted(self.regions.items(), key=lambda kv: kv[1][0]):
            if off < o:
                break
            last = (name, o, n)
        if last is None:
            return f'before first region (+{off})'
        name, o, n = last
        if off < o + n:
            return f'{name}+{off - o}'
        return f'gap after {name}+{off - o - n}' if off < self.cur else f'beyond arena (+{off - self.cur})'
