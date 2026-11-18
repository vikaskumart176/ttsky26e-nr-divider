"""Independent IEEE 754-2019 binary16 division reference (exact integer arithmetic, no float).

    ref16(a, b, mode) -> (q, flags)
        mode   0 = roundTiesToEven, 1 = roundTowardZero, 2 = roundTowardNegative, 3 = roundTowardPositive
        flags  {invalid, divide_by_zero, overflow, underflow, inexact} as bits 4..0
    Tininess is detected after rounding; NaN results are the canonical quiet NaN 0x7E00.
"""
import numpy as np
QNAN = 0x7E00


def _norm(m, e):
    for _ in range(11):
        need = (m > 0) & (((m >> 10) & 1) == 0)
        m = np.where(need, m << 1, m); e = e - need
    return m, e


def _inc(mode, s, lsb, g, st):
    if mode == 0:
        return g & (st | lsb)
    if mode == 1:
        return np.zeros_like(g)
    return (g | st) & (s if mode == 2 else 1 - s)


def prepare(a, b):
    a = np.asarray(a, dtype=np.int64) & 0xFFFF; b = np.asarray(b, dtype=np.int64) & 0xFFFF
    a, b = np.broadcast_arrays(a, b)
    sa, ea, fa = (a >> 15) & 1, (a >> 10) & 31, a & 0x3FF
    sb, eb, fb = (b >> 15) & 1, (b >> 10) & 31, b & 0x3FF
    s = sa ^ sb
    az, bz = (ea == 0) & (fa == 0), (eb == 0) & (fb == 0)
    ai, bi = (ea == 31) & (fa == 0), (eb == 31) & (fb == 0)
    an, bn = (ea == 31) & (fa != 0), (eb == 31) & (fb != 0)
    asn, bsn = an & ((fa >> 9) == 0), bn & ((fb >> 9) == 0)
    nan = an | bn | (az & bz) | (ai & bi)
    inv = (az & bz) | (ai & bi) | asn | bsn
    inf = ~nan & (ai | bz)
    dz = ~nan & bz & ~ai
    zer = ~nan & ~inf & (az | bi)
    fin = ~(nan | inf | zer)
    ma = np.where(fin, np.where(ea != 0, fa | 1024, fa), 1024)
    mb = np.where(fin, np.where(eb != 0, fb | 1024, fb), 1024)
    ma, ea1 = _norm(ma, np.maximum(ea, 1)); mb, eb1 = _norm(mb, np.maximum(eb, 1))
    lt = (ma < mb).astype(np.int64)
    n = ma << lt
    E = ea1 - eb1 + 15 - lt
    qi = (n << 12) // mb           # 13-bit quotient: leading 1, 10 fraction bits, guard, one more
    rem = (n << 12) % mb
    return dict(s=s, E=E, qi=qi, rem=rem, nan=nan, inv=inv, inf=inf, dz=dz, zer=zer, fin=fin)


def finish(p, mode):
    s, E, qi, rem = p['s'], p['E'], p['qi'], p['rem']
    # rounding with unbounded exponent, for tininess-after-rounding
    sig_u = qi >> 2; g_u = (qi >> 1) & 1; st_u = (qi & 1) | (rem != 0)
    m_u = sig_u + _inc(mode, s, sig_u & 1, g_u, st_u)
    # rounding at the real position (subnormal results shift right first)
    sh = np.clip(1 - E, 0, 40)
    tot = sh + 2
    kept = qi >> tot
    g = (qi >> (tot - 1)) & 1
    st = (((qi & ((1 << (tot - 1)) - 1)) != 0) | (rem != 0)).astype(np.int64)
    m = kept + _inc(mode, s, kept & 1, g, st)
    inexact = (g == 1) | (st == 1)
    normal = E >= 1
    carry = normal & ((m >> 11) == 1)
    m_n = np.where(carry, m >> 1, m); E_n = np.where(carry, E + 1, E)
    ovf = normal & (E_n >= 31)
    mag = np.where(normal, (np.clip(E_n, 0, 31) << 10) | (m_n & 0x3FF), m)
    to_inf = np.full_like(s, 1) if mode == 0 else (np.zeros_like(s) if mode == 1 else (s if mode == 2 else 1 - s))
    mag = np.where(ovf, np.where(to_inf == 1, 0x7C00, 0x7BFF), mag)
    inexact = inexact | ovf
    tiny = (E < 0) | ((E == 0) & (m_u != (1 << 11)))
    unf = tiny & inexact
    q = (s << 15) | mag
    q = np.where(p['zer'], s << 15, q); q = np.where(p['inf'], (s << 15) | 0x7C00, q); q = np.where(p['nan'], QNAN, q)
    fl = (p['inv'].astype(np.int64) << 4) | (p['dz'].astype(np.int64) << 3)
    fl = fl | np.where(p['fin'], (ovf.astype(np.int64) << 2) | (unf.astype(np.int64) << 1) | inexact.astype(np.int64), 0)
    return q, fl


def ref16(a, b, mode=0):
    return finish(prepare(a, b), mode)
