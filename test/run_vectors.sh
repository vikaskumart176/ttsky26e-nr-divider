#!/bin/bash
# Core-level regression without cocotb: the strict IEEE check against Berkeley
# SoftFloat / TestFloat (all four rounding modes), every normalised mantissa pair,
# and the fault-injection campaign on the self-check.
# Needs python3 + numpy, iverilog, and git + gcc + make (to build SoftFloat once).
# Add --exhaustive to check the model on every input pair (about 1 hour per core).
set -e
cd "$(dirname "$0")"
python3 ../model/verify_mantissa.py
python3 ../model/verify_softfloat.py "$@"
iverilog -o tb_vectors  tb_vectors.v  ../src/fp16_div_core.v
iverilog -o tb_mantissa tb_mantissa.v ../src/fp16_div_core.v
iverilog -o tb_fault    tb_fault.v    ../src/fp16_div_core.v
for f in testfloat_l1 softfloat; do
    echo "RTL vs $f.hex (expected values from SoftFloat)"
    vvp tb_vectors +VEC=$f.hex
done
python3 ../model/gen_mantissa_vectors.py mantissa.hex
vvp tb_mantissa
vvp tb_fault +VEC=finite.hex
