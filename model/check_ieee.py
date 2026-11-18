"""Check the golden model against the independent IEEE reference, in all 4 rounding modes.

    python check_ieee.py         quick: reference vs exact fractions, then the model on every
                                 mantissa pair, every special-value pair and 16M random pairs per mode
    python check_ieee.py full    every one of the 65536 x 65536 input pairs in every mode (~1-2 hours)
"""
import os, sys, time, math, random
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import fp16_div_model as K
import ieee16_ref as R

MODES = ["nearest-even", "toward zero", "down", "up"]


# ---- 1. slow exact-fraction oracle, used only to validate ieee16_ref ----
def _val(x):
    s = -1 if x >> 15 else 1; e = (x >> 10) & 31; f = x & 0x3FF
    if e == 31: return ('nan' if f else 'inf'), s
    return 'fin', s * (f if e == 0 else 1024 + f) * Fraction(2) ** (max(e, 1) - 25)

def _log2floor(x):
    e = math.floor(math.log2(x))
    while Fraction(2) ** e > x: e -= 1
    while Fraction(2) ** (e + 1) <= x: e += 1
    return e

def _round(x, mode, neg, unbounded=False):
    e = _log2floor(x) if unbounded else max(_log2floor(x), -14)
    ulp = Fraction(2) ** (e - 10); y = x / ulp; fl = math.floor(y); r = y - fl
    if r and (mode == 0 and (r > Fraction(1, 2) or (r == Fraction(1, 2) and fl % 2)) or
              mode in (2, 3) and neg == (mode == 2)):
        fl += 1
    return fl * ulp

def oracle(a, b, mode):
    (ka, va), (kb, vb) = _val(a), _val(b); s = (a ^ b) >> 15
    snan = lambda x: ((x >> 10) & 31) == 31 and (x & 0x3FF) and not (x >> 9) & 1
    if ka == 'nan' or kb == 'nan': return 0x7E00, 16 * bool(snan(a) or snan(b))
    if ka == 'inf' and kb == 'inf' or va == 0 and vb == 0 and ka == kb == 'fin': return 0x7E00, 16
    if ka == 'inf': return (s << 15) | 0x7C00, 0
    if kb == 'inf': return s << 15, 0
    if vb == 0: return (s << 15) | 0x7C00, 8
    if va == 0: return s << 15, 0
    x = abs(va / vb); neg = bool(s)
    r, ru = _round(x, mode, neg), _round(x, mode, neg, True)
    if ru > 65504:
        return (s << 15) | (0x7C00 if mode == 0 or (mode == 2) == neg and mode != 1 else 0x7BFF), 0b00101
    if r < Fraction(1, 2 ** 14): bits = int(r * 2 ** 24)
    else: e = _log2floor(r); bits = ((e + 15) << 10) | int(r / Fraction(2) ** (e - 10) - 1024)
    inexact = r != x
    return (s << 15) | bits, (2 if ru < Fraction(1, 2 ** 14) and inexact else 0) | inexact


def check_reference():
    rng = random.Random(5); c = []
    for _ in range(3000): c.append((rng.randrange(65536), rng.randrange(65536)))
    for _ in range(1500): c.append((rng.randrange(1024) | rng.randrange(2) << 15, rng.randrange(65536)))
    for _ in range(1500): c.append((rng.randrange(65536), rng.randrange(1024) | rng.randrange(2) << 15))
    rs = lambda lo, hi: (rng.randrange(2) << 15) | (rng.randrange(lo, hi) << 10) | rng.randrange(1024)
    for _ in range(2000): c.append((rs(1, 8), rs(18, 31)))
    for _ in range(2000): c.append((rs(20, 31), rs(1, 10)))
    sp = [0, 0x8000, 1, 0x8001, 0x3FF, 0x400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x7D00, 0x3C01, 0x3BFF]
    c += [(x, y) for x in sp for y in sp]
    A = np.array([x for x, _ in c]); B = np.array([y for _, y in c]); ok = True
    for m in range(4):
        q, f = R.ref16(A, B, m)
        bad = sum((int(q[i]), int(f[i])) != oracle(a, b, m) for i, (a, b) in enumerate(c))
        ok &= bad == 0
        print(f"  reference vs exact fractions, {MODES[m]:12s}: {len(c):,} pairs, {bad} mismatches")
    return ok


# ---- 2. golden model vs reference ----
isnan = lambda v: (((v >> 10) & 31) == 31) & ((v & 0x3FF) != 0)

def compare(A, B, tally):
    for m in range(4):
        qr, fr = R.ref16(A, B, m); qm, fm = K.fp16_div(A, B, m)
        bad = ~((qm == qr) | (isnan(qm) & isnan(qr))) | (fm != fr)
        tally[m][0] += A.size; tally[m][1] += int(bad.sum())
        if bad.any() and len(tally[4]) < 5:
            i = int(np.argmax(bad))
            tally[4].append(f"{MODES[m]} {int(A[i]):04X}/{int(B[i]):04X}: model {int(qm[i]):04X}/{int(fm[i]):05b} reference {int(qr[i]):04X}/{int(fr[i]):05b}")

def main(full):
    t0 = time.time()
    print("1. Is the IEEE reference itself right?")
    ok = check_reference()
    print("2. Golden model vs IEEE reference (result and all 5 flags)")
    tally = [[0, 0] for _ in range(4)] + [[]]
    ALL = np.arange(65536, dtype=np.int64)
    if full:
        for a0 in range(0, 65536, 32):
            compare(np.repeat(np.arange(a0, a0 + 32, dtype=np.int64), 65536), np.tile(ALL, 32), tally)
            if a0 % 4096 == 0: print(f"   A = 0x{a0:04X}  {time.time() - t0:.0f} s", flush=True)
    else:
        rng = np.random.default_rng(16)
        fa = np.repeat(np.arange(1024), 1024); fb = np.tile(np.arange(1024), 1024)
        sg = lambda n: rng.integers(0, 2, n) << 15
        compare(sg(fa.size) | 0x3C00 | fa, sg(fa.size) | 0x3C00 | fb, tally)        # every mantissa pair
        sp = np.array([0, 1, 0x3FF, 0x400, 0x3C00, 0x3C01, 0x7BFF, 0x7C00, 0x7E00, 0x7D00])
        sp = np.concatenate([sp, sp | 0x8000])
        compare(np.repeat(ALL, sp.size), np.tile(sp, 65536), tally)                 # anything / special
        compare(np.tile(sp, 65536), np.repeat(ALL, sp.size), tally)                 # special / anything
        for _ in range(4):
            compare(rng.integers(0, 65536, 4_000_000), rng.integers(0, 65536, 4_000_000), tally)
    for m in range(4):
        ok &= tally[m][1] == 0
        print(f"  model vs reference, {MODES[m]:12s}: {tally[m][0]:,} pairs, {tally[m][1]} mismatches")
    for e in tally[4]: print("    " + e)
    print(f"{'ALL CHECKS PASSED' if ok else 'FAILED'} in {time.time() - t0:.0f} s")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main(len(sys.argv) > 1 and sys.argv[1] == "full") else 1)
