/*
 * Copyright (c) 2026 Vikas, Shreekara, SreeHarsha, Girish, Kartik, Tharun
 * SPDX-License-Identifier: Apache-2.0
 *
 * FP16 Newton-Raphson divider - Tiny Tapeout wrapper.
 *
 * Host interface (byte-wide):
 *   ui_in[7:0]   data in
 *   uio_in[1:0]  address   write: 0=A[7:0] 1=A[15:8] 2=B[7:0] 3=B[15:8]
 *                          read:  0=Q[7:0] 1=Q[15:8] 2=flags  3=status
 *   uio_in[2]    write strobe (rising edge; addr/data must be stable while it is high)
 *   uio_in[3]    start (rising edge)
 *   uio_in[5:4]  rounding mode, sampled at start: 0 = nearest-even, 1 = toward zero,
 *                2 = down (toward -inf), 3 = up (toward +inf)
 *   uo_out[7:0]  data out (register selected by address)
 *   uio_out[7]   done (sticky until next start)
 *   uio_out[6]   busy
 *   flags byte = {fault, 2'b0, invalid, divzero, overflow, underflow, inexact}
 *                fault = built-in self-check failed (always 0 on working silicon)
 */
`default_nettype none

module tt_um_vikaskumart176_fp16_nr_div (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    // Only the two strobes are synchronised. Address and data are sampled once,
    // on the synchronised rising edge of the write strobe, which is one to two
    // clocks after the pin rose - inside the window where the host holds them stable.
    reg  [1:0] wr_sync, st_sync;
    reg        wr_prev, st_prev;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_sync <= 2'b00;
            st_sync <= 2'b00;
            wr_prev <= 1'b0;
            st_prev <= 1'b0;
        end else begin
            wr_sync <= {wr_sync[0], uio_in[2]};
            st_sync <= {st_sync[0], uio_in[3]};
            wr_prev <= wr_sync[1];
            st_prev <= st_sync[1];
        end
    end

    wire       wr      = wr_sync[1] & ~wr_prev;
    wire       start   = st_sync[1] & ~st_prev;
    wire [1:0] addr    = uio_in[1:0];

    reg  [15:0] a_reg, b_reg;
    reg         done_flag;
    wire [15:0] q;
    wire [4:0]  flags;
    wire        fault;
    wire        busy, done;

    // Operand registers have no reset: they are undefined until the host has written
    // all four bytes. The core reads a_reg / b_reg directly while busy, so they are
    // frozen from the start pulse until the divide finishes.
    always @(posedge clk) begin
        if (wr && !busy && !start) begin
            case (addr)
                2'd0: a_reg[7:0]  <= ui_in;
                2'd1: a_reg[15:8] <= ui_in;
                2'd2: b_reg[7:0]  <= ui_in;
                2'd3: b_reg[15:8] <= ui_in;
            endcase
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done_flag <= 1'b0;
        else if (start)
            done_flag <= 1'b0;
        else if (done)
            done_flag <= 1'b1;
    end

    fp16_div_core u_core (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (start),
        .a      (a_reg),
        .b      (b_reg),
        .rmode  (uio_in[5:4]),
        .result (q),
        .flags  (flags),
        .fault  (fault),
        .busy   (busy),
        .done   (done)
    );

    reg [7:0] dout;
    always @(*) begin
        case (addr)
            2'd0:    dout = q[7:0];
            2'd1:    dout = q[15:8];
            2'd2:    dout = {fault, 2'b00, flags};
            default: dout = {6'd0, busy, done_flag};
        endcase
    end

    assign uo_out  = dout;
    assign uio_out = {done_flag, busy, 6'd0};
    assign uio_oe  = 8'b1100_0000;

    wire _unused = &{ena, uio_in[7:6], 1'b0};

endmodule
