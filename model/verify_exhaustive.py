"""Exhaustive check of fp16_div_model against an independent reference.

Result bits: numpy float64 division, rounded to float16 by numpy (correctly rounded,
incl. subnormals; 53 >= 2*11+2 so double rounding is harmless).
Flags: derived from exact arithmetic, independent of the model's internals.
"""
import sys, time
import numpy as np
from fp16_div_model import fp16_div, QNAN

B = np.arange(65536, dtype=np.int64)
bf = B.astype(np.uint16).view(np.float16).astype(np.float64)
b_nan = np.isnan(bf)
b_snan = b_nan & (((B >> 9) & 1) == 0)
MIN_NORMAL = 2.0 ** -14


def ref(a_bits):
    af = np.float64(np.uint16(a_bits).view(np.float16))
    a_nan = np.isnan(af)
    a_snan = a_nan and ((a_bits >> 9) & 1) == 0
    with np.errstate(all="ignore"):
        q64 = af / bf
        r16 = q64.astype(np.float16)
    rbits = r16.view(np.uint16).astype(np.int64)
    rf = r16.astype(np.float64)

    invalid = ((af == 0) & (bf == 0)) | (np.isinf(af) & np.isinf(bf)) | a_snan | b_snan
    nan_res = np.isnan(q64) | a_nan | b_nan
    divz = (bf == 0) & np.isfinite(af) & (af != 0)
    finite_in = np.isfinite(af) & np.isfinite(bf) & (bf != 0) & (af != 0)
    with np.errstate(all="ignore"):
        inexact = finite_in & ~((rf * bf) == af)
    overflow = finite_in & np.isinf(rf)
    m, e = np.frexp(np.where(finite_in, q64, 1.0))
    r11 = np.ldexp(np.rint(m * 2048.0), e - 11)
    tiny = finite_in & (np.abs(r11) < MIN_NORMAL)
    underflow = tiny & inexact
    flags = (invalid * 16) | (divz * 8) | (overflow * 4) | (underflow * 2) | ((inexact | overflow) * 1)
    rbits = np.where(nan_res, QNAN, rbits)
    return rbits, flags.astype(np.int64)


def main(step=1):
    t0 = time.time()
    bad_r = bad_f = 0
    examples = []
    for a in range(0, 65536, step):
        rr, rf = ref(a)
        mr, mf = fp16_div(a, B)
        dr = rr != mr
        df = rf != mf
        bad_r += int(dr.sum()); bad_f += int(df.sum())
        if (dr | df).any() and len(examples) < 10:
            i = int(np.argmax(dr | df))
            examples.append((a, i, int(mr[i]), int(rr[i]), int(mf[i]), int(rf[i])))
        if a % 8192 == 0:
            print(f"  A=0x{a:04X}  elapsed {time.time()-t0:.0f}s  bad_result={bad_r} bad_flags={bad_f}", flush=True)
    total = (65536 // step) * 65536
    print(f"\nChecked {total:,} pairs in {time.time()-t0:.0f}s")
    print(f"Result mismatches: {bad_r}   Flag mismatches: {bad_f}")
    for a, b, mr, rr, mf, rf in examples:
        print(f"  A=0x{a:04X} B=0x{b:04X}: model 0x{mr:04X} f={mf:05b}  ref 0x{rr:04X} f={rf:05b}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
