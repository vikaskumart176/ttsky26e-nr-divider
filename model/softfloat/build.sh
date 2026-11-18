#!/bin/bash
# Fetch and build Berkeley SoftFloat 3e and TestFloat 3e (RISC-V specialisation),
# then the FP16 division oracles. Needs git, make and gcc. Run once; about a minute.
set -e
cd "$(dirname "$0")"
[ -d berkeley-softfloat-3 ] || git clone -q --depth 1 https://github.com/ucb-bar/berkeley-softfloat-3.git
[ -d berkeley-testfloat-3 ] || git clone -q --depth 1 https://github.com/ucb-bar/berkeley-testfloat-3.git
make -s -C berkeley-softfloat-3/build/Linux-x86_64-GCC SPECIALIZE_TYPE=RISCV
make -s -C berkeley-testfloat-3/build/Linux-x86_64-GCC SPECIALIZE_TYPE=RISCV SOFTFLOAT_DIR=../../../berkeley-softfloat-3
INC=berkeley-softfloat-3/source/include
LIB=berkeley-softfloat-3/build/Linux-x86_64-GCC/softfloat.a
gcc -O2 -I $INC sfdiv.c $LIB -o sfdiv
gcc -O2 -I $INC sfall.c $LIB -o sfall
echo "built: $(pwd)/sfdiv, $(pwd)/sfall and berkeley-testfloat-3/build/Linux-x86_64-GCC/testfloat_gen"
