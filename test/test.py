# SPDX-License-Identifier: Apache-2.0
import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from fp16_div_model import fp16_div  # noqa: E402
from ieee16_ref import ref16  # noqa: E402

WR, START = 0x04, 0x08


async def reset(dut):
    clock = Clock(dut.clk, 20, unit="ns")       # 50 MHz
    cocotb.start_soon(clock.start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def write_byte(dut, addr, data):
    dut.ui_in.value = data
    dut.uio_in.value = WR | addr
    await ClockCycles(dut.clk, 3)
    dut.uio_in.value = addr
    await ClockCycles(dut.clk, 1)


async def read_byte(dut, addr):
    dut.uio_in.value = addr
    await ClockCycles(dut.clk, 1)
    return int(dut.uo_out.value)


async def divide(dut, a, b, mode=0):
    await write_byte(dut, 0, a & 0xFF)
    await write_byte(dut, 1, a >> 8)
    await write_byte(dut, 2, b & 0xFF)
    await write_byte(dut, 3, b >> 8)
    dut.uio_in.value = START | (mode << 4)        # rounding mode on uio_in[5:4], sampled at start
    await ClockCycles(dut.clk, 3)
    dut.uio_in.value = (3 - mode) << 4             # change the pins: the core must keep the latched mode
    for _ in range(120):
        await ClockCycles(dut.clk, 1)
        if int(dut.uio_out.value) & 0x80:
            break
    else:
        raise AssertionError(f"timeout: a={a:04X} b={b:04X}")
    q = await read_byte(dut, 0) | (await read_byte(dut, 1) << 8)
    flags = await read_byte(dut, 2)
    return q, flags


async def check(dut, a, b, mode=0):
    q, flags = await divide(dut, a, b, mode)
    er, ef = fp16_div(a, b, mode)
    rr, rf = ref16(a, b, mode)
    er, ef, rr, rf = int(er), int(ef), int(rr), int(rf)
    assert (q, flags) == (er, ef), \
        f"mode {mode} {a:04X}/{b:04X}: got {q:04X} flags {flags:05b}, golden model {er:04X} flags {ef:05b}"
    assert (q, flags) == (rr, rf), \
        f"mode {mode} {a:04X}/{b:04X}: got {q:04X} flags {flags:05b}, IEEE reference {rr:04X} flags {rf:05b}"


CORNERS = [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x4000, 0x3555,
           0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x7D00, 0x0200, 0x1000, 0x6400]


@cocotb.test()
async def test_basic(dut):
    await reset(dut)
    dut._log.info("1.0 / 2.0 = 0.5")
    q, flags = await divide(dut, 0x3C00, 0x4000)
    assert q == 0x3800 and flags == 0
    dut._log.info("1.0 / 3.0 = 0x3555, inexact")
    q, flags = await divide(dut, 0x3C00, 0x4200)
    assert q == 0x3555 and flags == 0b00001


@cocotb.test()
async def test_corners(dut):
    await reset(dut)
    for a in CORNERS:
        for b in CORNERS:
            await check(dut, a, b)
    dut._log.info(f"{len(CORNERS) ** 2} corner-case pairs passed")


@cocotb.test()
async def test_random(dut):
    await reset(dut)
    rng = random.Random(2026)
    n = int(os.environ.get("N_RANDOM", "1000"))
    for _ in range(n):
        await check(dut, rng.randrange(65536), rng.randrange(65536))
    dut._log.info(f"{n} random pairs passed")


@cocotb.test()
async def test_subnormal(dut):
    await reset(dut)
    rng = random.Random(7)
    for _ in range(300):
        sub = rng.randrange(1, 1024) | (rng.randrange(2) << 15)
        other = rng.randrange(65536)
        await check(dut, sub, other)
        await check(dut, other, sub)
    dut._log.info("600 subnormal pairs passed")


MODES = ["nearest-even", "toward zero", "down", "up"]


@cocotb.test()
async def test_rounding_modes(dut):
    await reset(dut)
    # 1/3 and -1/3 land between two FP16 values, so each mode picks a side
    expect = {(0x3C00, 0x4200): [0x3555, 0x3555, 0x3555, 0x3556],
              (0xBC00, 0x4200): [0xB555, 0xB555, 0xB556, 0xB555],
              (0x7BFF, 0x0001): [0x7C00, 0x7BFF, 0x7BFF, 0x7C00],     # overflow: inf or largest finite
              (0xFBFF, 0x0001): [0xFC00, 0xFBFF, 0xFC00, 0xFBFF],
              (0x0001, 0x7BFF): [0x0000, 0x0000, 0x0000, 0x0001],     # underflow: 0 or smallest subnormal
              (0x8001, 0x7BFF): [0x8000, 0x8000, 0x8001, 0x8000]}
    for (a, b), qs in expect.items():
        for mode in range(4):
            q, _ = await divide(dut, a, b, mode)
            assert q == qs[mode], f"{a:04X}/{b:04X} {MODES[mode]}: got {q:04X}, expected {qs[mode]:04X}"
    dut._log.info("1/3, -1/3, overflow and underflow give the right side in all 4 modes")
    rng = random.Random(4)
    for mode in range(4):
        for a in CORNERS:
            for b in CORNERS:
                await check(dut, a, b, mode)
        for _ in range(300):
            await check(dut, rng.randrange(65536), rng.randrange(65536), mode)
        for _ in range(100):
            sub = rng.randrange(1, 1024) | (rng.randrange(2) << 15)
            await check(dut, sub, rng.randrange(65536), mode)
            await check(dut, rng.randrange(65536), sub, mode)
        dut._log.info(f"{MODES[mode]}: {len(CORNERS) ** 2 + 500} pairs match golden model and IEEE reference")
