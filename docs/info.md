## How it works

This project divides two IEEE 754 half-precision (FP16) numbers: Q = A / B. It supports the whole format: normal and subnormal numbers, +/-0, +/-Inf and NaN, round-to-nearest-even, and all five exception flags.

The mantissa division uses Newton-Raphson reciprocal iteration:

1. A 16-entry table, indexed by the top 4 fraction bits of B, gives a 6-bit first guess of 1/B.
2. Two iterations of `x = x * (2 - B * x)`, in 17-bit fixed-point arithmetic, refine the guess.
3. The quotient estimate is `q = A * x`.
4. The remainder `A - q * B` shows whether q is one too large or one too small, and whether any bits are left over (the sticky bit). The final rounding is therefore exact.

To fit in one tile, all multiplications share a single 20-bit adder and run as serial shift-and-add, one bit per clock. A division of normal numbers takes about 101 clock cycles (about 3 us at 33.3 MHz). Subnormal inputs or results need some extra shift cycles (126 at most). Special cases (zero, infinity, NaN, division by zero) finish in 1 cycle.

Verification: the bit-exact Python model was checked against an independent reference on all 2^32 operand pairs, and the RTL matches that model on 732,033 test vectors, including all corner cases.

### Registers

| Address `uio[1:0]` | Write     | Read   |
|--------------------|-----------|--------|
| 0                  | A[7:0]    | Q[7:0] |
| 1                  | A[15:8]   | Q[15:8]|
| 2                  | B[7:0]    | flags  |
| 3                  | B[15:8]   | status |

- flags = `{0, 0, 0, invalid, divide_by_zero, overflow, underflow, inexact}`
- status = `{0, 0, 0, 0, 0, 0, busy, done}`

The result Q is written over A, and B is also changed during the calculation, so write all four operand bytes before every division.

Special cases:

- 0/0, Inf/Inf, or a signalling NaN input: result NaN (0x7E00), invalid flag set.
- Any other NaN input: result 0x7E00, no flags.
- Finite non-zero number divided by zero: result +/-Inf, divide-by-zero flag set.
- Result too large: +/-Inf with the overflow and inexact flags.

## How to test

All control inputs are synchronised inside the design, so they can be driven slowly, for example from the microcontroller on the demo board.

To write one byte:

1. Put the address on `uio[1:0]` and the byte on `ui_in`.
2. Set `uio[2]` (WR) high for at least 4 clock cycles, then low for at least 4. Keep the address and data stable while WR is high.

To run one division:

1. Write A low, A high, B low and B high (addresses 0 to 3).
2. Set `uio[3]` (START) high for at least 4 clock cycles, then low.
3. Wait for `uio[7]` (DONE) to go high. It stays high until the next START. `uio[6]` (BUSY) is high while the divider is working.
4. Set `uio[1:0]` to 0 and then 1 to read the low and high bytes of Q on `uo_out`, and to 2 to read the flags.

Examples:

| A              | B              | Q              | flags |
|----------------|----------------|----------------|-------|
| 0x3C00 (1.0)   | 0x4000 (2.0)   | 0x3800 (0.5)   | 0x00  |
| 0x3C00 (1.0)   | 0x4200 (3.0)   | 0x3555 (0.333) | 0x01 (inexact) |
| 0x3C00 (1.0)   | 0x0000 (0.0)   | 0x7C00 (+Inf)  | 0x08 (divide by zero) |
| 0x0000 (0.0)   | 0x0000 (0.0)   | 0x7E00 (NaN)   | 0x10 (invalid) |
| 0x7BFF (65504) | 0x3800 (0.5)   | 0x7C00 (+Inf)  | 0x05 (overflow, inexact) |

## External hardware

None. The demo board's microcontroller (or any other) drives the inputs and reads the outputs.
