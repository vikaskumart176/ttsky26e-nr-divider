#include <stdio.h>
#include <stdint.h>
#include "softfloat.h"
/*
 * FP16 division oracle on Berkeley SoftFloat 3e (RISC-V specialisation:
 * canonical NaN 0x7E00, tininess detected after rounding).
 * stdin : "AAAA BBBB M" per line, M = 0 nearest-even, 1 toward zero,
 *         2 toward -inf, 3 toward +inf (SoftFloat's own numbering)
 * stdout: "RRRR FF", FF = {invalid, divzero, overflow, underflow, inexact}
 *         (SoftFloat's flag bits are already in this order)
 */
int main(void){
  unsigned a,b,m; char line[128];
  while (fgets(line,sizeof line,stdin)) {
    if (sscanf(line,"%x %x %u",&a,&b,&m)!=3) continue;
    float16_t fa,fb,q; fa.v=a; fb.v=b;
    softfloat_roundingMode=(uint_fast8_t)m; softfloat_exceptionFlags=0;
    q=f16_div(fa,fb);
    printf("%04X %02X\n",(unsigned)q.v,(unsigned)softfloat_exceptionFlags);
  }
  return 0;
}
