/*
 * Copyright (c) 2026 Vikas, Shreekara, SreeHarsha, Girish, Kartik, Tharun
 * SPDX-License-Identifier: Apache-2.0
 *
 * FP16 Newton-Raphson divider (1-tile version) - Tiny Tapeout wrapper.
 *
 * Host interface (byte-wide):
 *   ui_in[7:0]   data in
 *   uio_in[1:0]  address   write: 0=A[7:0] 1=A[15:8] 2=B[7:0] 3=B[15:8]
 *                          read:  0=Q[7:0] 1=Q[15:8] 2=flags  3=status
 *   uio_in[2]    write strobe: data is sampled once, on its synchronised rising edge.
 *                Keep address and data stable while the strobe is high (>= 4 clocks).
 *   uio_in[3]    start (rising edge)
 *   uo_out[7:0]  data out (register selected by address)
 *   uio_out[7]   done (sticky until next start)
 *   uio_out[6]   busy
 *   flags  = {3'b0, invalid, divzero, overflow, underflow, inexact}
 *   status = {6'b0, busy, done}
 *   The result Q overwrites A: write all four operand bytes before every start.
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

    reg [1:0] wr_sync, st_sync;
    reg       wr_prev, st_prev, done_flag;

    wire wr_pulse = wr_sync[1] & ~wr_prev;
    wire start    = st_sync[1] & ~st_prev;
    wire busy, done;
    wire [7:0] rd_data;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_sync   <= 2'b00;
            st_sync   <= 2'b00;
            wr_prev   <= 1'b0;
            st_prev   <= 1'b0;
            done_flag <= 1'b0;
        end else begin
            wr_sync <= {wr_sync[0], uio_in[2]};
            st_sync <= {st_sync[0], uio_in[3]};
            wr_prev <= wr_sync[1];
            st_prev <= st_sync[1];
            if (start)
                done_flag <= 1'b0;
            else if (done)
                done_flag <= 1'b1;
        end
    end

    fp16_div_core u_core (
        .clk     (clk),
        .rst_n   (rst_n),
        .wr_en   (wr_pulse),
        .wr_addr (uio_in[1:0]),
        .wr_data (ui_in),
        .start   (start),
        .rd_addr (uio_in[1:0]),
        .rd_data (rd_data),
        .busy    (busy),
        .done    (done)
    );

    assign uo_out  = (uio_in[1:0] == 2'd3) ? {6'd0, busy, done_flag} : rd_data;
    assign uio_out = {done_flag, busy, 6'd0};
    assign uio_oe  = 8'b1100_0000;

    wire _unused = &{ena, uio_in[7:4], 1'b0};

endmodule
