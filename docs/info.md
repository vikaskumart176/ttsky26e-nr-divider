<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This project is an IEEE 754 **half-precision (FP16)** floating-point divider that computes **Q = A / B** using a
**Newton-Raphson reciprocal**.

- **Format:** binary16 (1 sign, 5 exponent, 10 fraction bits)
- **Rounding:** all four IEEE 754 rounding modes, selected per divide: nearest-even, toward zero,
  down (toward -inf), up (toward +inf); every result is correctly rounded
- **Subnormals:** fully supported on inputs and outputs
- **Exception flags:** invalid, divide-by-zero, overflow, underflow, inexact
- **NaN results:** canonical quiet NaN `0x7E00`; underflow uses tininess-after-rounding
- **Built-in self-check:** a `fault` flag that catches hardware upsets in the reciprocal and quotient
  computation, at almost no extra hardware

**Algorithm**

1. Unpack A and B; normalise subnormal mantissas; detect zero, infinity and NaN.
2. Seed the reciprocal of B's mantissa from a 16-entry, 6-bit lookup table. The seed is rounded down so it
   never exceeds the true reciprocal.
3. Two Newton-Raphson iterations in residual form, `x = x * (1 + e)` with `e = 1 - B * x`, in 15-bit fixed
   point. `e` is taken as the bitwise complement of the product bits, so no subtractor is needed, and `x`
   always approaches the reciprocal from below.
4. Quotient estimate `q1 = floor(A * x) + 1`. Because `x` never overshoots, `q1` is either the exact
   quotient or one more than it.
5. Remainder multiply `q1 * B`, preloaded with the constant `2^11 - 1`. The true remainder `r` lies in
   `[-B, B)`, so the low 12 bits of the product give the sign and zero-ness of `r` directly, and the quotient
   is corrected by a conditional decrement. No remainder register or subtractor is needed.
6. Normalise, denormalise for subnormal results, round in the selected mode, pack, and set the flags.
   On overflow the result is infinity or the largest finite number, as the mode requires.

**Built-in self-check**

Because of the preload, a correct estimate makes the remainder product equal `A * 2^13 + (2^11 - 1 - r)`.
Its upper bits must therefore be exactly `2 * A`, and its low bits must lie within `+/- B`. This holds only if
`q1` is the true quotient or one more, so any upset in the reciprocal iterations or the quotient multiply that
would give a wrong result breaks the pattern and sets the `fault` bit (bit 7 of the flags byte). The check
reuses the multiply the divider needs anyway: it costs a comparator and one flag register, and no extra cycles.

In a fault-injection campaign (one random bit flip in M, H, L or the recode carry at a random clock of each of
19,977 divides), 9,991 upsets were detected, 9,912 were masked (the result was still correct) and 74 caused a
silent error. All 74 silent cases hit the final remainder/rounding step, which the check does not claim to
cover; no upset in the reciprocal or quotient multiplies went undetected.

**Datapath**

The whole computation runs on **three working registers**: a multiplicand `M`, an accumulator `H` and a
shift register `L` that holds the scanned operand and the low product bits. The reciprocal, residual,
quotient and remainder are never stored separately; each one is read out of `{H, L}` at the end of one
multiply and handed to the next. The multiplier is radix-4 (2 bits per clock) with digit 3 recoded as
4 - 1, so it needs no 3x multiple.

The widths (15-bit reciprocal, 16-bit residual, 6-bit seed) are the smallest for which the result is
still correctly rounded. This was established by exhaustive search over all 1,048,576 pairs of
normalised mantissas (`model/verify_mantissa.py`).

A divide takes **49 clock cycles**. A power-of-two divisor (mantissa exactly 1.0) skips the reciprocal and
takes 2 cycles; special cases such as 0, infinity or NaN take 1. The design is clocked at **25 MHz**, so
one divide takes about 2 us.

**Verification**

- `model/verify_softfloat.py` checks the golden model against **Berkeley SoftFloat 3e** (the reference
  software implementation of IEEE 754, RISC-V conventions: canonical NaN, tininess after rounding). With
  `--exhaustive` it covers **every one of the 4,294,967,296 input pairs in all four rounding modes**
  (17.2 billion divisions, results and all five flags).
- The RTL is run against vector files whose expected values come from SoftFloat itself, including
  TestFloat's official `f16_div` level-1 and level-2 suites, and against every one of the 1,048,576
  normalised mantissa pairs (`test/tb_mantissa.v`). The self-check must stay 0 on every vector.
- `model/check_ieee.py` also checks the model against an independent exact-integer IEEE reference.
- The cocotb tests in `test/test.py` drive the full chip interface in all four rounding modes.

## How to test

The chip uses a byte-wide register interface.

| Pin | Function |
|---|---|
| `ui_in[7:0]` | data byte to write |
| `uio_in[1:0]` | address. Write: 0 = A[7:0], 1 = A[15:8], 2 = B[7:0], 3 = B[15:8]. Read: 0 = Q[7:0], 1 = Q[15:8], 2 = flags, 3 = status |
| `uio_in[2]` | write strobe, rising edge (hold address and data stable while it is high) |
| `uio_in[3]` | start (rising edge) |
| `uio_in[5:4]` | rounding mode, sampled at start: 0 = nearest-even, 1 = toward zero, 2 = down (toward -inf), 3 = up (toward +inf) |
| `uo_out[7:0]` | byte selected by the address |
| `uio_out[6]` | busy |
| `uio_out[7]` | done (stays high until the next start) |

Flags byte: `{fault, 0, 0, invalid, divzero, overflow, underflow, inexact}`. `fault` is the self-check result
and reads 0 in normal operation. Status byte: `{0, 0, 0, 0, 0, 0, busy, done}`.

**To perform one division:**

1. Reset the design (`rst_n` low, then high) and provide a clock (up to 25 MHz).
2. For each of the four bytes (A low, A high, B low, B high): put the byte on `ui_in`, set `uio_in[1:0]` to its
   address, raise `uio_in[2]` for at least 3 clock cycles, then lower it.
3. Set the rounding mode on `uio_in[5:4]` (0 for the usual round-to-nearest-even), then pulse `uio_in[3]`
   (start) high for at least 3 clock cycles, keeping the mode stable while start is high.
4. Wait until `uio_out[7]` (done) is high. Do not write operands while `uio_out[6]` (busy) is high.
5. Set `uio_in[1:0]` to 0, 1 and 2 and read Q low, Q high and the flags on `uo_out`.

**Example values**

| A | B | Q | Flags | Meaning |
|---|---|---|---|---|
| `3C00` (1.0) | `4000` (2.0) | `3800` (0.5) | `00000` | exact |
| `3C00` (1.0) | `4200` (3.0) | `3555` (0.33325) | `00001` | inexact |
| `3C00` (1.0) | `0000` (0.0) | `7C00` (+inf) | `01000` | divide-by-zero |
| `0000` (0.0) | `0000` (0.0) | `7E00` (NaN) | `10000` | invalid |
| `7BFF` (65504) | `0001` (min subnormal) | `7C00` (+inf) | `00101` | overflow + inexact |

The same divisions in the other rounding modes (Q for modes 0 / 1 / 2 / 3):

| A | B | Q in mode 0 / 1 / 2 / 3 |
|---|---|---|
| `3C00` (1.0) | `4200` (3.0) | `3555` / `3555` / `3555` / `3556` |
| `BC00` (-1.0) | `4200` (3.0) | `B555` / `B555` / `B556` / `B555` |
| `7BFF` (65504) | `0001` | `7C00` / `7BFF` / `7BFF` / `7C00` (overflow: inf or largest finite) |
| `0001` | `7BFF` | `0000` / `0000` / `0000` / `0001` (underflow: zero or smallest subnormal) |

## External hardware

None. The Tiny Tapeout demo board's microcontroller can drive the inputs and read the outputs directly.
