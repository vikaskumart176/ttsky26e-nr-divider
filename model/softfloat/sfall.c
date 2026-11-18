#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "softfloat.h"
/*
 * Exhaustive FP16 division table on Berkeley SoftFloat 3e, for every dividend a in
 * [a0, a1), every rounding mode 0..3 and every divisor b: writes 65536 uint16
 * results, then 65536 uint8 flags, per (a, mode) block, to stdout (binary).
 * Usage: sfall a0 a1
 */
int main(int argc, char **argv){
  if (argc != 3) return 1;
  unsigned a0 = (unsigned) strtoul(argv[1], 0, 0), a1 = (unsigned) strtoul(argv[2], 0, 0);
  static uint16_t r[65536]; static uint8_t f[65536];
  for (unsigned a = a0; a < a1; a++)
    for (unsigned m = 0; m < 4; m++) {
      softfloat_roundingMode = (uint_fast8_t) m;
      for (unsigned b = 0; b < 65536; b++) {
        float16_t fa, fb, q; fa.v = (uint16_t) a; fb.v = (uint16_t) b;
        softfloat_exceptionFlags = 0;
        q = f16_div(fa, fb);
        r[b] = q.v; f[b] = (uint8_t) softfloat_exceptionFlags;
      }
      fwrite(r, 2, 65536, stdout); fwrite(f, 1, 65536, stdout);
    }
  return 0;
}
