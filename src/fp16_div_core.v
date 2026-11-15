/*
 * FP16 (IEEE 754 binary16) divider, Newton-Raphson reciprocal - 1-tile version.
 * Round-to-nearest-even, full subnormal support, all five IEEE flags.
 * Area-optimised: radix-2 sequential multiplier, bit-serial (de)normalisation,
 * operand registers reused for working values and the result.
 * Bit-exact with model/fp16_div_model.py.
 *
 * Write port: wr_en (1-cycle pulse), wr_addr 0..3 = A lo, A hi, B lo, B hi.
 * Read port : rd_addr 0..3 = Q lo, Q hi, flags, status (combinational).
 * The result Q overwrites A; write all four bytes before every start.
 */
`default_nettype none

module fp16_div_core (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       wr_en,
    input  wire [1:0] wr_addr,
    input  wire [7:0] wr_data,
    input  wire       start,
    input  wire [1:0] rd_addr,
    output reg  [7:0] rd_data,
    output wire       busy,
    output reg        done
);

    localparam [4:0] S_IDLE = 5'd0,
                     S_NIN  = 5'd1,   // normalise subnormal inputs, then seed
                     S_MBX  = 5'd2,   // multiply mb * x
                     S_T    = 5'd3,   // t = 2 - mb*x
                     S_MXT  = 5'd4,   // multiply x * t
                     S_XU   = 5'd5,   // x = x*t
                     S_MAX  = 5'd6,   // multiply ma * x
                     S_QU   = 5'd7,   // q = ma*x
                     S_MQB  = 5'd8,   // multiply q * mb
                     S_R    = 5'd9,   // r = -(q*mb) mod 2^13
                     S_C1   = 5'd10,  // compare r with 0 and mb
                     S_CNEG = 5'd11,  // r < 0: sticky from r + mb
                     S_CDEC = 5'd12,  // q = q - 1
                     S_CINC = 5'd13,  // q = q + 1
                     S_NORM = 5'd14,  // normalise q
                     S_TINY = 5'd15,  // tininess (after rounding)
                     S_DEN  = 5'd16,  // denormalise for subnormal results
                     S_RND  = 5'd17;  // round, pack, flags

    reg  [4:0]  state;
    reg  [15:0] ra, rb;        // operands; ra[10:0]/rb[10:0] become mantissas, ra becomes Q
    reg  signed [7:0] E;
    reg  [16:0] x;
    reg  [17:0] tq;            // t during NR, then q
    reg  [35:0] pr;            // {H[18:0], L[16:0]} multiplier accumulator, holds r afterwards
    reg  [4:0]  cnt;
    reg  [4:0]  flags;
    reg         stk, iter, tiny, subn;

    assign busy = (state != S_IDLE);

    // ---------------- special cases (decoded from raw operands at start) ----------------
    wire [4:0] ea = ra[14:10], eb = rb[14:10];
    wire [9:0] fa = ra[9:0],   fb = rb[9:0];
    wire a_zero = (ea == 5'd0)  && (fa == 10'd0);
    wire b_zero = (eb == 5'd0)  && (fb == 10'd0);
    wire a_inf  = (ea == 5'd31) && (fa == 10'd0);
    wire b_inf  = (eb == 5'd31) && (fb == 10'd0);
    wire a_nan  = (ea == 5'd31) && (fa != 10'd0);
    wire b_nan  = (eb == 5'd31) && (fb != 10'd0);
    wire invalid_n = (a_zero && b_zero) || (a_inf && b_inf) ||
                     (a_nan && !fa[9]) || (b_nan && !fb[9]);
    wire nan_n     = a_nan || b_nan || invalid_n;
    wire divz_n    = b_zero && !a_zero && !a_inf && !a_nan;
    wire inf_n     = (a_inf && !b_inf && !b_nan) || divz_n;
    wire zero_n    = (a_zero && !b_zero && !b_nan && !a_nan) || (b_inf && !a_inf && !a_nan);
    wire s_n       = ra[15] ^ rb[15];
    wire [4:0] ea1 = (ea == 5'd0) ? 5'd1 : ea;
    wire [4:0] eb1 = (eb == 5'd0) ? 5'd1 : eb;

    function [5:0] seed;
        input [3:0] i;
        begin
            case (i)
                4'd0:  seed = 6'd62; 4'd1:  seed = 6'd59; 4'd2:  seed = 6'd55; 4'd3:  seed = 6'd53;
                4'd4:  seed = 6'd50; 4'd5:  seed = 6'd48; 4'd6:  seed = 6'd46; 4'd7:  seed = 6'd44;
                4'd8:  seed = 6'd42; 4'd9:  seed = 6'd40; 4'd10: seed = 6'd39; 4'd11: seed = 6'd37;
                4'd12: seed = 6'd36; 4'd13: seed = 6'd35; 4'd14: seed = 6'd34; default: seed = 6'd33;
            endcase
        end
    endfunction

    wire [10:0] ma = ra[10:0];
    wire [10:0] mb = rb[10:0];

    // ---------------- radix-2 multiplier + main adder ----------------
    // Multiplicand selects x or tq; scanned operand (loaded into L) selects mb, x or ma.
    wire        mul_state = (state == S_MBX) || (state == S_MXT) || (state == S_MAX) || (state == S_MQB);
    wire        n17       = (state == S_MXT);
    wire [17:0] mcand     = (state == S_MXT || state == S_MQB) ? tq : {1'b0, x};
    wire [16:0] mscan     = (state == S_MXT) ? x : (state == S_MAX) ? {6'd0, ma} : {6'd0, mb};

    // main adder: accumulate, or two's-complement negate (t and r)
    reg  [19:0] add_a, add_b;
    reg         add_c;
    always @(*) begin
        add_a = 20'd0; add_b = 20'd0; add_c = 1'b0;
        case (state)
            S_T: begin add_b = {2'b11, ~pr[33:16]}; add_c = 1'b1; end         // -(mb*x >> 10)
            S_R: begin add_b = {7'h7F, ~pr[18:6]};  add_c = 1'b1; end         // -(q*mb) low 13 bits
            default: begin
                add_a = {1'b0, pr[35:17]};
                add_b = pr[0] ? {2'b00, mcand} : 20'd0;
            end
        endcase
    end
    wire [19:0] add_s = add_a + add_b + {19'd0, add_c};

    // ---------------- normalise / tininess / rounding inputs ----------------
    wire        nrm     = ~tq[13];
    wire [10:0] keep_u  = tq[13:3];
    wire        inc_u   = tq[2] & ((|tq[1:0]) | stk | keep_u[0]);
    wire        carry_u = inc_u & (&keep_u);

    wire        rnd     = tq[2];
    wire        stk2    = (|tq[1:0]) | stk;
    wire        inexact = rnd | stk2;
    wire        rnd_inc = rnd & (stk2 | tq[3]);

    // ---------------- small adder: remainder correction, q +/- 1, rounding ----------------
    wire signed [13:0] r13  = {pr[12], pr[12:0]};
    reg  [14:0] sa_a, sa_b;
    reg         sa_c;
    always @(*) begin
        sa_a = 15'd0; sa_b = 15'd0; sa_c = 1'b0;
        case (state)
            S_C1:   begin sa_a = {r13[13], r13}; sa_b = ~{4'd0, mb}; sa_c = 1'b1; end   // r - mb
            S_CNEG: begin sa_a = {r13[13], r13}; sa_b = {4'd0, mb};               end   // r + mb
            S_CDEC: begin sa_a = tq[14:0];       sa_b = 15'h7FFF;                 end   // q - 1
            S_CINC: begin sa_a = tq[14:0];       sa_c = 1'b1;                     end   // q + 1
            S_RND:  begin sa_a = {4'd0, tq[13:3]}; sa_c = rnd_inc;                end   // keep + inc
            default: ;
        endcase
    end
    wire [14:0] sa_s = sa_a + sa_b + {14'd0, sa_c};

    // ---------------- round / pack ----------------
    wire [11:0] keep2   = sa_s[11:0];
    wire signed [7:0] Eb0 = subn ? 8'sd0 : E;
    wire signed [7:0] Er  = keep2[11] ? (Eb0 + 8'sd1) : (subn && keep2[10]) ? 8'sd1 : Eb0;
    wire [9:0]  frac    = keep2[11] ? keep2[10:1] : keep2[9:0];
    wire        ovf     = (Er >= 8'sd31);

    // ---------------- read port ----------------
    always @(*) begin
        case (rd_addr)
            2'd0:    rd_data = ra[7:0];
            2'd1:    rd_data = ra[15:8];
            2'd2:    rd_data = {3'b000, flags};
            default: rd_data = {6'd0, busy, 1'b0};
        endcase
    end

    // ---------------- FSM ----------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; ra <= 16'd0; rb <= 16'd0; E <= 8'sd0;
            x <= 17'd0; tq <= 18'd0; pr <= 36'd0; cnt <= 5'd0; flags <= 5'd0;
            stk <= 1'b0; iter <= 1'b0; tiny <= 1'b0; subn <= 1'b0; done <= 1'b0;
        end else begin
            done <= 1'b0;

            if (mul_state) begin
                if (cnt == 5'd0) begin
                    pr  <= {19'd0, mscan};
                    cnt <= n17 ? 5'd17 : 5'd11;
                end else begin
                    pr  <= {add_s, pr[16:1]};
                    cnt <= cnt - 5'd1;
                end
            end

            case (state)
                S_IDLE: begin
                    if (wr_en) begin
                        case (wr_addr)
                            2'd0: ra[7:0]  <= wr_data;
                            2'd1: ra[15:8] <= wr_data;
                            2'd2: rb[7:0]  <= wr_data;
                            2'd3: rb[15:8] <= wr_data;
                        endcase
                    end
                    if (start) begin
                        if (nan_n || inf_n || zero_n) begin
                            ra    <= nan_n ? 16'h7E00 : inf_n ? {s_n, 15'h7C00} : {s_n, 15'd0};
                            flags <= {invalid_n, divz_n, 3'b000};
                            done  <= 1'b1;
                        end else begin
                            ra    <= {s_n, 4'd0, (ea != 5'd0), fa};
                            rb    <= {5'd0, (eb != 5'd0), fb};
                            E     <= $signed({3'd0, ea1}) - $signed({3'd0, eb1}) + 8'sd15;
                            stk   <= 1'b0;
                            state <= S_NIN;
                        end
                    end
                end
                S_NIN: begin
                    if (ra[10] && rb[10]) begin
                        x     <= {1'b0, seed(rb[9:6]), 10'd0};
                        iter  <= 1'b0;
                        state <= S_MBX;
                    end else begin
                        if (!ra[10]) ra[10:0] <= {ra[9:0], 1'b0};
                        if (!rb[10]) rb[10:0] <= {rb[9:0], 1'b0};
                        E <= E + (rb[10] ? 8'sd0 : 8'sd1) - (ra[10] ? 8'sd0 : 8'sd1);
                    end
                end
                S_MBX: if (cnt == 5'd1) state <= S_T;
                S_T: begin
                    tq    <= {1'b0, add_s[16:0]};
                    state <= S_MXT;
                end
                S_MXT: if (cnt == 5'd1) state <= S_XU;
                S_XU: begin
                    x     <= pr[32:16];
                    iter  <= 1'b1;
                    state <= iter ? S_MAX : S_MBX;
                end
                S_MAX: if (cnt == 5'd1) state <= S_QU;
                S_QU: begin
                    tq    <= {3'd0, pr[33:19]};
                    state <= S_MQB;
                end
                S_MQB: if (cnt == 5'd1) state <= S_R;
                S_R: begin
                    pr[12:0] <= add_s[12:0];
                    state    <= S_C1;
                end
                S_C1: begin
                    if (r13 < 0)
                        state <= S_CNEG;
                    else if (!sa_s[14]) begin               // r - mb >= 0
                        stk   <= (sa_s[13:0] != 14'd0);
                        state <= S_CINC;
                    end else begin
                        stk   <= (r13 != 14'sd0);
                        state <= S_NORM;
                    end
                end
                S_CNEG: begin
                    stk   <= (sa_s[13:0] != 14'd0);
                    state <= S_CDEC;
                end
                S_CDEC: begin
                    tq[14:0] <= sa_s;
                    state    <= S_NORM;
                end
                S_CINC: begin
                    tq[14:0] <= sa_s;
                    state    <= S_NORM;
                end
                S_NORM: begin
                    if (nrm) begin
                        tq[13:0] <= {tq[12:0], 1'b0};
                        E        <= E - 8'sd1;
                    end
                    state <= S_TINY;
                end
                S_TINY: begin
                    tiny  <= (E < 0) || ((E == 0) && !carry_u);
                    subn  <= (E <= 0);
                    state <= S_DEN;
                end
                S_DEN: begin
                    if ((E < 8'sd1) && (tq[13:0] != 14'd0)) begin
                        tq[13:0] <= {1'b0, tq[13:1]};
                        stk      <= stk | tq[0];
                        E        <= E + 8'sd1;
                    end else
                        state <= S_RND;
                end
                S_RND: begin
                    if (ovf) begin
                        ra    <= {ra[15], 15'h7C00};
                        flags <= 5'b00101;
                    end else begin
                        ra    <= {ra[15], Er[4:0], frac};
                        flags <= {3'b000, tiny & inexact, inexact};
                    end
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
