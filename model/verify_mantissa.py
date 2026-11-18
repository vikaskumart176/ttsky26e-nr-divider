"""Exhaustive proof of the minimum-precision quotient datapath.

Checks nr_quotient against exact integer division for every pair of normalised
11-bit mantissas (1024 x 1024), plus the range invariants the RTL relies on, and
that the built-in self-check never raises a false alarm on a fault-free datapath.
Run:  python verify_mantissa.py
"""
import numpy as np
from fp16_div_model import nr_quotient, nr_core, SEED_LUT, F, TE, P

MA = np.arange(1024, 2048, dtype=np.int64)[:, None]
MB = np.arange(1024, 2048, dtype=np.int64)[None, :]


def main():
    q, rem_nz = nr_quotient(MA, MB)
    Q = (MA << P) // MB
    R = (MA << P) - Q * MB
    bad_q = int((q != Q).sum())
    bad_r = int((rem_nz != (R != 0)).sum())

    # invariants: b*x < 1 and x < 1 at every step (so the residual is unsigned)
    mb = MB[0]
    x = SEED_LUT[(mb - 1024) >> 6] << (F - 6)
    ok = True
    for _ in range(2):
        pr = x * mb
        ok &= bool((pr < (1 << (F + 10))).all())
        e = ~(pr >> (F + 10 - TE)) & ((1 << TE) - 1)
        x = (x * ((1 << TE) + e)) >> TE
        ok &= bool((x < (1 << F)).all())
    q1 = (MA * x[None, :] + (1 << (F + 10 - P))) >> (F + 10 - P)
    d = q1 - Q
    ok &= bool(((d == 0) | (d == 1)).all())

    _, _, chk = nr_core(MA, MB)
    false_alarm = int((~chk).sum())

    print(f"mantissa pairs checked : {Q.size:,}")
    print(f"quotient mismatches    : {bad_q}")
    print(f"sticky mismatches      : {bad_r}")
    print(f"range invariants hold  : {ok}")
    print(f"self-check false alarms: {false_alarm}")
    print(f"estimate q1 == Q in {int((d == 0).sum()):,} pairs, Q + 1 in {int((d == 1).sum()):,}")
    return bad_q == 0 and bad_r == 0 and ok and false_alarm == 0


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
