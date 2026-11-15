"""Bit-accurate golden model of the FP16 Newton-Raphson divider.

Every integer width and truncation here matches the planned RTL, so this
model doubles as the reference for the cocotb testbench. Works on numpy
uint16 arrays (vectorised) or plain ints.
"""
import numpy as np

F = 16          # fractional bits of the reciprocal x
P = 13          # quotient bits below the binary point before normalisation
QNAN = 0x7E00   # canonical quiet NaN

# 16-entry seed table: 1/midpoint of each mantissa interval, 6 fractional bits
SEED_LUT = np.array([62, 59, 55, 53, 50, 48, 46, 44, 42, 40, 39, 37, 36, 35, 34, 33],
                    dtype=np.int64)

FLAG_INVALID, FLAG_DIVZERO, FLAG_OVERFLOW, FLAG_UNDERFLOW, FLAG_INEXACT = 16, 8, 4, 2, 1


def clz11(f):
    """Leading zeros of an 11-bit value (f > 0)."""
    n = np.zeros_like(f)
    for bit in range(10, -1, -1):
        n = np.where(f >= (1 << bit), n, n + 1)
        f = np.where(f >= (1 << bit), f | (1 << 20), f)   # latch once found
    return n


def unpack(x):
    s = (x >> 15) & 1
    e = (x >> 10) & 0x1F
    f = x & 0x3FF
    is_zero = (e == 0) & (f == 0)
    is_sub = (e == 0) & (f != 0)
    is_inf = (e == 31) & (f == 0)
    is_nan = (e == 31) & (f != 0)
    is_snan = is_nan & ((f >> 9) == 0)
    lz = np.where(is_sub, clz11(np.where(is_sub, f, 1)), 0)
    m = np.where(is_sub, f << lz, f | 1024)                 # normalised 11-bit mantissa
    E = np.where(is_sub, 1 - lz, e)                         # biased exponent, may be <= 0
    return s, E, m, is_zero, is_inf, is_nan, is_snan


def nr_quotient(ma, mb):
    """floor(ma * 2^P / mb) and nonzero-remainder flag, via seed + 2 NR steps + correction."""
    x = SEED_LUT[(mb - 1024) >> 6] << (F - 6)
    for _ in range(2):
        bx = (mb * x) >> 10                 # mul: 11 x 17
        x = (x * ((2 << F) - bx)) >> F      # mul: 17 x 18
    q = (ma * x) >> (F + 10 - P)            # mul: 11 x 17
    r = (ma << P) - q * mb                  # mul: 15 x 11
    lo = r < 0
    hi = r >= mb
    q = np.where(lo, q - 1, np.where(hi, q + 1, q))
    r = np.where(lo, r + mb, np.where(hi, r - mb, r))
    return q, r != 0


def fp16_div(a, b):
    """Returns (result_bits, flags) for FP16 a / b, RNE, full subnormals, tininess after rounding."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    a, b = np.broadcast_arrays(a, b)
    sa, Ea, ma, za, ia, na, sna = unpack(a)
    sb, Eb, mb, zb, ib, nb, snb = unpack(b)
    s = sa ^ sb

    q, rem_nz = nr_quotient(ma, mb)
    E = Ea - Eb + 15
    norm = (q >> P) == 0                    # quotient < 1 -> shift left
    q = np.where(norm, q << 1, q)           # now leading 1 at bit P (14-bit value)
    E = np.where(norm, E - 1, E)

    # Tininess after rounding: round to 11 bits with unbounded exponent, check E stays <= 0
    keep_u = q >> 3
    rnd_u = (q >> 2) & 1
    stk_u = ((q & 3) != 0) | rem_nz
    inc_u = rnd_u & (stk_u | (keep_u & 1))
    carry_u = (keep_u + inc_u) >> 11
    tiny = (E + carry_u) <= 0

    # Denormalise for subnormal results (shift right into exponent 0)
    sh = np.clip(1 - E, 0, 15)
    shifted_out = (q & ((1 << sh) - 1)) != 0
    qd = q >> sh
    sticky_extra = shifted_out
    E_res = np.where(E <= 0, 0, E)

    keep = qd >> 3
    rnd = (qd >> 2) & 1
    stk = ((qd & 3) != 0) | rem_nz | sticky_extra
    inexact = (rnd == 1) | stk
    keep = keep + (rnd & (stk | (keep & 1)))

    # Rounding carry: normal 2048 -> renormalise; subnormal 1024 -> becomes smallest normal
    ovf_m = keep >> 11
    keep = np.where(ovf_m == 1, keep >> 1, keep)
    E_res = np.where(ovf_m == 1, E_res + 1, E_res)
    E_res = np.where((E <= 0) & ((keep >> 10) == 1), 1, E_res)

    overflow = E_res >= 31
    res = (s << 15) | (E_res << 10) | (keep & 0x3FF)
    res = np.where(overflow, (s << 15) | 0x7C00, res)
    underflow = tiny & inexact
    flags = (overflow * (FLAG_OVERFLOW | FLAG_INEXACT)) | (underflow * FLAG_UNDERFLOW) | (inexact * FLAG_INEXACT)

    # Special cases override the arithmetic path
    invalid = (za & zb) | (ia & ib) | sna | snb
    any_nan = na | nb | invalid
    divz = zb & ~za & ~ia & ~na
    inf_res = (ia & ~ib & ~nb) | divz
    zero_res = (za & ~zb & ~nb & ~na) | (ib & ~ia & ~na)

    res = np.where(zero_res, s << 15, res)
    res = np.where(inf_res, (s << 15) | 0x7C00, res)
    res = np.where(any_nan, QNAN, res)
    special = any_nan | inf_res | zero_res
    flags = np.where(special, 0, flags)
    flags = flags | (invalid * FLAG_INVALID) | (divz * FLAG_DIVZERO)
    return res.astype(np.int64), flags.astype(np.int64)


if __name__ == "__main__":
    tests = [(0x3C00, 0x4000, "1/2"), (0x4200, 0x3C00, "3/1"), (0x3C00, 0x4200, "1/3"),
             (0x0001, 0x3C00, "min_sub/1"), (0x7BFF, 0x0001, "max/min_sub"),
             (0x0000, 0x0000, "0/0"), (0x3C00, 0x0000, "1/0"), (0x7C00, 0x7C00, "inf/inf")]
    for a, b, name in tests:
        r, f = fp16_div(a, b)
        ref = (np.uint16(a).view(np.float16).astype(np.float64) /
               np.uint16(b).view(np.float16).astype(np.float64))
        print(f"{name:12s} 0x{int(r):04X}  flags={int(f):05b}  ({np.uint16(int(r)).view(np.float16)} vs numpy {ref})")
