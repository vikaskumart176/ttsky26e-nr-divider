"""Strict IEEE 754 check against Berkeley SoftFloat 3e and the TestFloat suites.

SoftFloat is the reference software implementation of IEEE 754 (by John Hauser,
used to verify the RISC-V and many commercial FPUs). It is built here with the
RISC-V specialisation: canonical quiet NaN 0x7E00 for every NaN result, and
tininess detected after rounding - the same conventions as this divider.

  1. model : fp16_div_model.fp16_div against SoftFloat in all four rounding modes on
             every mantissa pair, every special-value pair and random operand classes.
             With --exhaustive: every one of the 2^32 input pairs in every mode
             (17.2 billion divisions; about 1 hour per core, --jobs N to split it).
  2. files : vectors whose expected values come from SoftFloat, not from the model,
             for the RTL testbench (test/tb_vectors.v) and the fault campaign:
               test/testfloat_l1.hex TestFloat's official f16_div level-1 suite, 4 modes
               test/testfloat_l2.hex level 2 (about 5 M vectors) with --level2
               test/softfloat.hex    operand classes x 4 modes
               test/finite.hex       finite non-zero operands for test/tb_fault.v

Usage: python verify_softfloat.py [--exhaustive] [--jobs N] [--level2]
"""
import os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from fp16_div_model import fp16_div

HERE = os.path.dirname(os.path.abspath(__file__))
SF = os.path.join(HERE, "softfloat")
SFDIV = os.path.join(SF, "sfdiv")
SFALL = os.path.join(SF, "sfall")
TFGEN = os.path.join(SF, "berkeley-testfloat-3", "build", "Linux-x86_64-GCC", "testfloat_gen")
TEST = os.path.join(HERE, "..", "test")
MODES = ["near_even", "minMag", "min", "max"]          # TestFloat names of modes 0..3
B = np.arange(65536, dtype=np.int64)


def softfloat(A, B_, M):
    inp = "".join(f"{a:04X} {b:04X} {m}\n" for a, b, m in zip(A.tolist(), B_.tolist(), M.tolist()))
    out = subprocess.run([SFDIV], input=inp.encode(), capture_output=True, check=True).stdout.split()
    return (np.array([int(x, 16) for x in out[0::2]], dtype=np.int64),
            np.array([int(x, 16) for x in out[1::2]], dtype=np.int64))


def classes(rng, n):
    """Operand classes, n pairs each."""
    u = lambda: rng.integers(0, 1 << 16, n, dtype=np.int64)
    def fp(e, f=None):
        f = rng.integers(0, 1 << 10, n, dtype=np.int64) if f is None else f
        return (rng.integers(0, 2, n, dtype=np.int64) << 15) | (e << 10) | f
    ea = rng.integers(0, 32, n, dtype=np.int64)
    eb = rng.integers(0, 32, n, dtype=np.int64)
    z = np.zeros(n, dtype=np.int64)
    yield "uniform bits", u(), u()
    yield "all exponent pairs", fp(ea), fp(eb)
    yield "subnormal / anything", fp(z), u()
    yield "anything / subnormal", u(), fp(z)
    yield "subnormal / subnormal", fp(z), fp(z)
    eb2 = rng.integers(1, 31, n, dtype=np.int64)
    yield "underflow boundary", fp(np.clip(eb2 - 15 + rng.integers(-12, 4, n), 0, 30)), fp(eb2)
    yield "overflow boundary", fp(np.clip(eb2 + 16 + rng.integers(-3, 4, n), 0, 30)), fp(eb2)
    yield "power-of-two divisors", fp(ea), fp(eb, z)


def model(A, B_, M):
    """fp16_div with a per-vector rounding mode (the model takes one mode per call)."""
    R = np.zeros_like(A); F = np.zeros_like(A)
    for m in range(4):
        k = M == m
        if k.any():
            R[k], F[k] = fp16_div(A[k], B_[k], m)
    return R, F


def write_hex(path, A, B_, M, R, F):
    with open(path, "w") as f:
        for a, b, m, r, g in zip(A.tolist(), B_.tolist(), M.tolist(), R.tolist(), F.tolist()):
            f.write(f"{a:04x}{b:04x}{m:02x}{r:04x}{g:02x}\n")


def testfloat(level, path):
    n = 0
    with open(path, "w") as f:
        for m, name in enumerate(MODES):
            out = subprocess.run([TFGEN, "-level", str(level), "-r" + name, "-tininessafter", "f16_div"],
                                 capture_output=True, check=True, text=True).stdout
            for line in out.splitlines():
                a, b, r, g = line.split()
                f.write(f"{a.lower()}{b.lower()}{m:02x}{r.lower()}{g.lower()}\n")
                n += 1
    print(f"TestFloat f16_div level {level}: {n:,} vectors (4 modes) -> test/{os.path.basename(path)}")


def exhaustive_slice(a_lo, a_hi):
    """Model vs SoftFloat for every dividend in [a_lo, a_hi), every divisor, every mode."""
    bad = [0, 0, 0, 0]; example = None
    for a0 in range(a_lo, a_hi, 64):
        a1 = min(a0 + 64, a_hi)
        raw = subprocess.run([SFALL, str(a0), str(a1)], capture_output=True, check=True).stdout
        for i, a in enumerate(range(a0, a1)):
            A = np.full(65536, a, dtype=np.int64)
            for m in range(4):
                off = (i * 4 + m) * 65536 * 3
                sr = np.frombuffer(raw, dtype=np.uint16, count=65536, offset=off).astype(np.int64)
                sf = np.frombuffer(raw, dtype=np.uint8, count=65536, offset=off + 131072).astype(np.int64)
                mr, mf = fp16_div(A, B, m)
                d = (mr != sr) | (mf != sf)
                if d.any():
                    bad[m] += int(d.sum())
                    if example is None:
                        k = int(np.argmax(d))
                        example = (a, k, m, int(mr[k]), int(mf[k]), int(sr[k]), int(sf[k]))
    return a_lo, a_hi, bad, example


def main():
    args = sys.argv[1:]
    jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 1
    if not (os.path.exists(SFDIV) and os.path.exists(SFALL) and os.path.exists(TFGEN)):
        subprocess.run([os.path.join(SF, "build.sh")], check=True)
    t0 = time.time()
    rng = np.random.default_rng(1985)

    # every mantissa pair (exponents 15/15), all modes, both signs of the result
    fa = np.repeat(np.arange(1024, dtype=np.int64), 1024)
    fb = np.tile(np.arange(1024, dtype=np.int64), 1024)
    groups = [("every mantissa pair", 0x3C00 | fa, (rng.integers(0, 2, fb.size) << 15) | 0x3C00 | fb)]
    # every special-value pair
    sp = np.array([0x0000, 0x8000, 0x0001, 0x83FF, 0x0400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00,
                   0x7E00, 0xFE00, 0x7C01, 0x7D00, 0x3C00, 0xBC00, 0x3555], dtype=np.int64)
    groups.append(("special-value pairs", np.repeat(sp, sp.size), np.tile(sp, sp.size)))
    for _ in range(8):
        groups += list(classes(rng, 250_000))
    total = 0; bad = np.zeros(4, dtype=np.int64)
    for name, A, B_ in groups:
        A4, B4 = np.repeat(A, 4), np.repeat(B_, 4)
        M4 = np.tile(np.arange(4, dtype=np.int64), A.size)
        mr, mf = model(A4, B4, M4)
        sr, sf = softfloat(A4, B4, M4)
        d = (mr != sr) | (mf != sf)
        total += A4.size
        np.add.at(bad, M4[d], 1)
        if d.any():
            k = int(np.argmax(d))
            print(f"  MISMATCH [{name}] mode {M4[k]} A=0x{A4[k]:04X} B=0x{B4[k]:04X}: "
                  f"model 0x{mr[k]:04X} f={mf[k]:05b}  softfloat 0x{sr[k]:04X} f={sf[k]:05b}")
    print(f"Model vs Berkeley SoftFloat: {total:,} vectors (all 4 modes), mismatches {int(bad.sum())}  "
          f"(per mode {bad.tolist()})  {time.time() - t0:.0f}s")
    ok = bad.sum() == 0

    if "--exhaustive" in args:
        cuts = np.linspace(0, 65536, 65, dtype=np.int64)          # 64 slices of 1024 dividends
        done = 0; ex_bad = np.zeros(4, dtype=np.int64); example = None
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for lo, hi, b, e in pool.map(exhaustive_slice, cuts[:-1].tolist(), cuts[1:].tolist()):
                done += (hi - lo) * 65536 * 4
                ex_bad += np.array(b); example = example or e
                print(f"  exhaustive: {done:,} / 17,179,869,184 divisions, mismatches {int(ex_bad.sum())}"
                      f"  {time.time() - t0:.0f}s", flush=True)
        print(f"Exhaustive model vs Berkeley SoftFloat: 17,179,869,184 divisions (every input pair, "
              f"all 4 modes), mismatches {int(ex_bad.sum())} (per mode {ex_bad.tolist()})")
        if example:
            print(f"  first mismatch: a=0x{example[0]:04X} b=0x{example[1]:04X} mode {example[2]}: model "
                  f"0x{example[3]:04X} f={example[4]:05b}  softfloat 0x{example[5]:04X} f={example[6]:05b}")
        ok = ok and ex_bad.sum() == 0

    # vector files for the RTL, expected values from SoftFloat
    testfloat(1, os.path.join(TEST, "testfloat_l1.hex"))
    if "--level2" in args:
        testfloat(2, os.path.join(TEST, "testfloat_l2.hex"))
    A, B_ = zip(*[(a, b) for _, a, b in classes(np.random.default_rng(2024), 8_000)])
    A, B_ = np.concatenate(A), np.concatenate(B_)
    A4, B4 = np.repeat(A, 4), np.repeat(B_, 4)
    M4 = np.tile(np.arange(4, dtype=np.int64), A.size)
    write_hex(os.path.join(TEST, "softfloat.hex"), A4, B4, M4, *softfloat(A4, B4, M4))
    print(f"SoftFloat operand classes: {A4.size:,} vectors (4 modes) -> test/softfloat.hex")
    r = np.random.default_rng(2026)
    e = r.integers(0, 31, (2, 20_000), dtype=np.int64); f = r.integers(0, 1 << 10, (2, 20_000), dtype=np.int64)
    f = np.where((e == 0) & (f == 0), 1, f)
    ops = (r.integers(0, 2, (2, 20_000), dtype=np.int64) << 15) | (e << 10) | f
    M = r.integers(0, 4, 20_000, dtype=np.int64)
    write_hex(os.path.join(TEST, "finite.hex"), ops[0], ops[1], M, *softfloat(ops[0], ops[1], M))
    print("Finite operands for the fault campaign: 20,000 vectors -> test/finite.hex")
    print(f"Done in {time.time() - t0:.0f}s")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
