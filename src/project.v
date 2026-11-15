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
 *   uio_in[2]    write strobe (level; addr/data must be stable while it is high)
 *   uio_in[3]    start (rising edge)
 *   uo_out[7:0]  data out (register selected by address)
 *   uio_out[7]   done (sticky until next start)
 *   uio_out[6]   busy
 *   flags byte = {3'b0, invalid, divzero, overflow, underflow, inexact}
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

    reg  [1:0] wr_sync, st_sync;
    reg  [1:0] addr_s1, addr_s2;
    reg  [7:0] data_s1, data_s2;
    reg        st_prev;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_sync <= 2'b00;
            st_sync <= 2'b00;
            st_prev <= 1'b0;
            addr_s1 <= 2'd0;
            addr_s2 <= 2'd0;
            data_s1 <= 8'd0;
            data_s2 <= 8'd0;
        end else begin
            wr_sync <= {wr_sync[0], uio_in[2]};
            st_sync <= {st_sync[0], uio_in[3]};
            st_prev <= st_sync[1];
            addr_s1 <= uio_in[1:0];
            addr_s2 <= addr_s1;
            data_s1 <= ui_in;
            data_s2 <= data_s1;
        end
    end

    wire       wr      = wr_sync[1];
    wire       start   = st_sync[1] & ~st_prev;
    wire [1:0] waddr   = addr_s2;
    wire [1:0] addr    = uio_in[1:0];

    reg  [15:0] a_reg, b_reg;
    reg         done_flag;
    wire [15:0] q;
    wire [4:0]  flags;
    wire        busy, done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg     <= 16'd0;
            b_reg     <= 16'd0;
            done_flag <= 1'b0;
        end else begin
            if (wr && !busy) begin
                case (waddr)
                    2'd0: a_reg[7:0]  <= data_s2;
                    2'd1: a_reg[15:8] <= data_s2;
                    2'd2: b_reg[7:0]  <= data_s2;
                    2'd3: b_reg[15:8] <= data_s2;
                endcase
            end
            if (start)
                done_flag <= 1'b0;
            else if (done)
                done_flag <= 1'b1;
        end
    end

    fp16_div_core u_core (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (start),
        .a      (a_reg),
        .b      (b_reg),
        .result (q),
        .flags  (flags),
        .busy   (busy),
        .done   (done)
    );

    reg [7:0] dout;
    always @(*) begin
        case (addr)
            2'd0:    dout = q[7:0];
            2'd1:    dout = q[15:8];
            2'd2:    dout = {3'b000, flags};
            default: dout = {6'd0, busy, done_flag};
        endcase
    end

    assign uo_out  = dout;
    assign uio_out = {done_flag, busy, 6'd0};
    assign uio_oe  = 8'b1100_0000;

    wire _unused = &{ena, uio_in[7:4], 1'b0};

endmodule
