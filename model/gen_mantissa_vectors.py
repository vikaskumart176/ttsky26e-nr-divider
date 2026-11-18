"""Write expected {flags, result} for every normalised mantissa pair (exponent 0),
one 21-bit hex word per line, for test/tb_mantissa.v.
Usage: python gen_mantissa_vectors.py out.hex
"""
import sys
import numpy as np
from fp16_div_model import fp16_div

fa = np.arange(1024, dtype=np.int64)[:, None]
fb = np.arange(1024, dtype=np.int64)[None, :]
res, flags = fp16_div(0x3C00 | fa, 0x3C00 | fb)
with open(sys.argv[1], "w") as f:
    for w in ((flags << 16) | res).ravel():
        f.write(f"{int(w):06x}\n")
