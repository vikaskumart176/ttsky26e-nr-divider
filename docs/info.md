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
- **Rounding:** round-to-nearest, ties-to-even (correctly rounded)
- **Subnormals:** fully supported on inputs and outputs
- **Exception flags:** invalid, divide-by-zero, overflow, underflow, inexact
- **NaN results:** canonical quiet NaN `0x7E00`; underflow uses tininess-after-rounding

**Algorithm**

1. Unpack A and B; normalise subnormal mantissas; detect zero, infinity and NaN.
2. Seed the reciprocal of B's mantissa from a 16-entry lookup table.
3. Two Newton-Raphson iterations, `x = x * (2 - B * x)`, in 17-bit fixed point.
4. Quotient `q = A * x`, then a remainder check `r = A - q * B` corrects q by +/-1 so the quotient bits are exact.
5. Normalise, denormalise for subnormal results, round to nearest even, pack, and set the flags.

All multiplications share **one radix-4 sequential multiplier** (2 bits per clock). A divide takes
**58 clock cycles** (special cases such as 0, infinity or NaN take 1). The design is clocked at **33.33 MHz**,
so one divide takes about 1.7 us.

The design was verified bit-exactly against a Python golden model, which itself was checked against an
independent reference on all 4,294,967,296 input pairs.

## How to test

The chip uses a byte-wide register interface.

| Pin | Function |
|---|---|
| `ui_in[7:0]` | data byte to write |
| `uio_in[1:0]` | address. Write: 0 = A[7:0], 1 = A[15:8], 2 = B[7:0], 3 = B[15:8]. Read: 0 = Q[7:0], 1 = Q[15:8], 2 = flags, 3 = status |
| `uio_in[2]` | write strobe (hold address and data stable while it is high) |
| `uio_in[3]` | start (rising edge) |
| `uo_out[7:0]` | byte selected by the address |
| `uio_out[6]` | busy |
| `uio_out[7]` | done (stays high until the next start) |

Flags byte: `{0, 0, 0, invalid, divzero, overflow, underflow, inexact}`. Status byte: `{0, 0, 0, 0, 0, 0, busy, done}`.

**To perform one division:**

1. Reset the design (`rst_n` low, then high) and provide a clock (up to 33 MHz).
2. For each of the four bytes (A low, A high, B low, B high): put the byte on `ui_in`, set `uio_in[1:0]` to its
   address, raise `uio_in[2]` for at least 3 clock cycles, then lower it.
3. Pulse `uio_in[3]` (start) high for at least 3 clock cycles.
4. Wait until `uio_out[7]` (done) is high.
5. Set `uio_in[1:0]` to 0, 1 and 2 and read Q low, Q high and the flags on `uo_out`.

**Example values**

| A | B | Q | Flags | Meaning |
|---|---|---|---|---|
| `3C00` (1.0) | `4000` (2.0) | `3800` (0.5) | `00000` | exact |
| `3C00` (1.0) | `4200` (3.0) | `3555` (0.33325) | `00001` | inexact |
| `3C00` (1.0) | `0000` (0.0) | `7C00` (+inf) | `01000` | divide-by-zero |
| `0000` (0.0) | `0000` (0.0) | `7E00` (NaN) | `10000` | invalid |
| `7BFF` (65504) | `0001` (min subnormal) | `7C00` (+inf) | `00101` | overflow + inexact |

## External hardware

None. The Tiny Tapeout demo board's microcontroller can drive the inputs and read the outputs directly.
