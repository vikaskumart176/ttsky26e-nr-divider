/*
 * FP16 (IEEE 754 binary16) divider, Newton-Raphson reciprocal.
 * Round-to-nearest-even, full subnormal support, all five IEEE flags.
 * One shared radix-4 sequential multiplier (2 multiplier bits per cycle),
 * sequenced by an FSM (~58 cycles per divide).
 * Bit-exact with model/fp16_div_model.py.
 */
`default_nettype none

module fp16_div_core (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg  [15:0] result,
    output reg  [4:0]  flags,      // {invalid, divzero, overflow, underflow, inexact}
    output wire        busy,
    output reg         done
);

    localparam [3:0] S_IDLE  = 4'd0,
                     S_NR_A  = 4'd1,   // multiply mb*x (6 steps)
                     S_NR_AU = 4'd2,   // t = 2 - mb*x
                     S_NR_B  = 4'd3,   // multiply x*t  (9 steps)
                     S_NR_BU = 4'd4,   // x = x*t
                     S_QUO   = 4'd5,   // multiply ma*x (6 steps)
                     S_QUOU  = 4'd6,   // q = ma*x
                     S_REM   = 4'd7,   // multiply q*mb (6 steps)
                     S_REMU  = 4'd8,   // r = ma<<13 - q*mb
                     S_CORR  = 4'd9,
                     S_NORM  = 4'd10,
                     S_ROUND = 4'd11;

    reg [3:0] state;
    assign busy = (state != S_IDLE);

    // ---------------- unpack (combinational on inputs) ----------------
    wire        sa = a[15], sb = b[15];
    wire [4:0]  ea = a[14:10], eb = b[14:10];
    wire [9:0]  fa = a[9:0],  fb = b[9:0];

    wire a_zero = (ea == 5'd0)  && (fa == 10'd0);
    wire b_zero = (eb == 5'd0)  && (fb == 10'd0);
    wire a_sub  = (ea == 5'd0)  && (fa != 10'd0);
    wire b_sub  = (eb == 5'd0)  && (fb != 10'd0);
    wire a_inf  = (ea == 5'd31) && (fa == 10'd0);
    wire b_inf  = (eb == 5'd31) && (fb == 10'd0);
    wire a_nan  = (ea == 5'd31) && (fa != 10'd0);
    wire b_nan  = (eb == 5'd31) && (fb != 10'd0);
    wire a_snan = a_nan && !fa[9];
    wire b_snan = b_nan && !fb[9];

    function [3:0] clz10;          // leading zeros of a 10-bit value, as an 11-bit field (>= 1)
        input [9:0] f;
        begin
            casez (f)
                10'b1?????????: clz10 = 4'd1;
                10'b01????????: clz10 = 4'd2;
                10'b001???????: clz10 = 4'd3;
                10'b0001??????: clz10 = 4'd4;
                10'b00001?????: clz10 = 4'd5;
                10'b000001????: clz10 = 4'd6;
                10'b0000001???: clz10 = 4'd7;
                10'b00000001??: clz10 = 4'd8;
                10'b000000001?: clz10 = 4'd9;
                default:        clz10 = 4'd10;
            endcase
        end
    endfunction

    wire [3:0]  lza = clz10(fa);
    wire [3:0]  lzb = clz10(fb);
    wire [10:0] ma_n = a_sub ? ({1'b0, fa} << lza) : {1'b1, fa};
    wire [10:0] mb_n = b_sub ? ({1'b0, fb} << lzb) : {1'b1, fb};
    wire signed [7:0] Ea_n = a_sub ? (8'sd1 - $signed({4'd0, lza})) : $signed({3'd0, ea});
    wire signed [7:0] Eb_n = b_sub ? (8'sd1 - $signed({4'd0, lzb})) : $signed({3'd0, eb});

    wire invalid_n = (a_zero && b_zero) || (a_inf && b_inf) || a_snan || b_snan;
    wire nan_n     = a_nan || b_nan || invalid_n;
    wire divz_n    = b_zero && !a_zero && !a_inf && !a_nan;
    wire inf_n     = (a_inf && !b_inf && !b_nan) || divz_n;
    wire zero_n    = (a_zero && !b_zero && !b_nan && !a_nan) || (b_inf && !a_inf && !a_nan);
    wire special_n = nan_n || inf_n || zero_n;
    wire s_n       = sa ^ sb;

    // seed table: 1/midpoint of mantissa interval, 6 fractional bits
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

    // ---------------- datapath registers ----------------
    reg               s;
    reg        [10:0] ma, mb;
    reg signed [7:0]  E;
    reg        [16:0] x;
    reg        [17:0] t;
    reg        [14:0] q;
    reg signed [26:0] r;
    reg               rem_nz;
    reg               iter;
    reg        [13:0] qn;
    reg               tiny;

    // ---------------- shared radix-4 sequential multiplier ----------------
    // pr = {H[20:0], L[17:0]}: L starts as the scanned operand B, H accumulates.
    // Each step adds dig*A (dig = L[1:0], 0..3) to H and shifts {sum, L} right by 2.
    // After n steps the product is pr >> (18 - 2n).
    reg  [17:0] mul_a;               // multiplicand
    reg  [17:0] mul_b;               // scanned operand
    reg         mul_n9;              // 1: 9 steps (18-bit B), 0: 6 steps (12-bit B)
    reg  [38:0] pr;
    reg  [3:0]  cnt;
    reg         mrun;

    always @(*) begin
        case (state)
            S_NR_A:  begin mul_a = {1'b0, x};     mul_b = {7'd0, mb}; mul_n9 = 1'b0; end
            S_NR_B:  begin mul_a = t;             mul_b = {1'b0, x};  mul_n9 = 1'b1; end
            S_QUO:   begin mul_a = {1'b0, x};     mul_b = {7'd0, ma}; mul_n9 = 1'b0; end
            S_REM:   begin mul_a = {3'd0, q};     mul_b = {7'd0, mb}; mul_n9 = 1'b0; end
            default: begin mul_a = 18'd0;         mul_b = 18'd0;      mul_n9 = 1'b0; end
        endcase
    end

    wire [1:0]  dig   = pr[1:0];
    wire [20:0] a1    = {3'd0, mul_a};
    wire [20:0] a2    = {2'd0, mul_a, 1'b0};
    wire [20:0] pp    = (dig == 2'd0) ? 21'd0 :
                        (dig == 2'd1) ? a1 :
                        (dig == 2'd2) ? a2 : (a1 + a2);
    wire [21:0] psum  = {1'b0, pr[38:18]} + {1'b0, pp};
    wire [39:0] pnext = {psum, pr[17:0]} >> 2;
    wire [34:0] prod6 = {2'b00, pr[38:6]};   // product after 6 steps
    wire [34:0] prod9 = pr[34:0];            // product after 9 steps

    wire        m_last = mrun && (cnt == 4'd1);

    // ---------------- normalise / round (combinational) ----------------
    wire               nrm     = ~q[13];
    wire        [13:0] qn_n    = nrm ? {q[12:0], 1'b0} : q[13:0];
    wire signed [7:0]  En      = E - {7'd0, nrm};
    wire        [10:0] keep_u  = qn_n[13:3];
    wire               inc_u   = qn_n[2] & ((|qn_n[1:0]) | rem_nz | keep_u[0]);
    wire               carry_u = inc_u & (&keep_u);
    wire               tiny_n  = (En < 0) || ((En == 0) && !carry_u);

    wire               subn    = (E <= 0);                    // E holds En in S_ROUND
    wire signed [7:0]  sh_s    = 8'sd1 - E;
    wire        [3:0]  sh      = !subn ? 4'd0 : (sh_s > 8'sd15) ? 4'd15 : sh_s[3:0];
    wire        [13:0] qd      = qn >> sh;
    wire        [13:0] lost    = qn & ~({14{1'b1}} << sh);
    wire        [10:0] keep    = qd[13:3];
    wire               rnd     = qd[2];
    wire               stk     = (|qd[1:0]) | rem_nz | (|lost);
    wire               inexact = rnd | stk;
    wire        [11:0] keep2   = {1'b0, keep} + {11'd0, rnd & (stk | keep[0])};
    wire signed [7:0]  Eb0     = subn ? 8'sd0 : E;
    wire signed [7:0]  Er      = keep2[11] ? (Eb0 + 8'sd1) :
                                 (subn && keep2[10]) ? 8'sd1 : Eb0;
    wire        [9:0]  frac    = keep2[11] ? keep2[10:1] : keep2[9:0];
    wire               ovf     = (Er >= 8'sd31);
    wire               unf     = tiny & inexact;

    // ---------------- FSM ----------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            result <= 16'd0;
            flags  <= 5'd0;
            done   <= 1'b0;
            s <= 1'b0; ma <= 11'd0; mb <= 11'd0; E <= 8'sd0;
            x <= 17'd0; t <= 18'd0; q <= 15'd0; r <= 27'sd0;
            rem_nz <= 1'b0; iter <= 1'b0; qn <= 14'd0; tiny <= 1'b0;
            pr <= 39'd0; cnt <= 4'd0; mrun <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    s  <= s_n;
                    ma <= ma_n;
                    mb <= mb_n;
                    E  <= Ea_n - Eb_n + 8'sd15;
                    x  <= {1'b0, seed(mb_n[9:6]), 10'd0};
                    iter <= 1'b0;
                    if (special_n) begin
                        result <= nan_n ? 16'h7E00 :
                                  inf_n ? {s_n, 15'h7C00} : {s_n, 15'd0};
                        flags  <= {invalid_n, divz_n, 3'b000};
                        done   <= 1'b1;
                    end else
                        state <= S_NR_A;
                end
                S_NR_A, S_NR_B, S_QUO, S_REM: begin
                    if (!mrun) begin
                        pr   <= {21'd0, mul_b};
                        cnt  <= mul_n9 ? 4'd9 : 4'd6;
                        mrun <= 1'b1;
                    end else begin
                        pr   <= pnext[38:0];
                        cnt  <= cnt - 4'd1;
                        if (m_last) begin
                            mrun  <= 1'b0;
                            state <= state + 4'd1;          // to the matching *U state
                        end
                    end
                end
                S_NR_AU: begin
                    t     <= 18'd131072 - prod6[27:10];         // 2 - b*x
                    state <= S_NR_B;
                end
                S_NR_BU: begin
                    x     <= prod9[32:16];                      // x*(2 - b*x)
                    iter  <= 1'b1;
                    state <= iter ? S_QUO : S_NR_A;
                end
                S_QUOU: begin
                    q     <= prod6[27:13];                      // a*x
                    state <= S_REM;
                end
                S_REMU: begin
                    r     <= $signed({3'b000, ma, 13'd0}) - $signed({1'b0, prod6[25:0]});
                    state <= S_CORR;
                end
                S_CORR: begin
                    if (r < 0) begin
                        q <= q - 15'd1;  rem_nz <= (r + $signed({16'd0, mb})) != 0;
                    end else if (r >= $signed({16'd0, mb})) begin
                        q <= q + 15'd1;  rem_nz <= (r - $signed({16'd0, mb})) != 0;
                    end else
                        rem_nz <= (r != 0);
                    state <= S_NORM;
                end
                S_NORM: begin
                    qn    <= qn_n;
                    E     <= En;
                    tiny  <= tiny_n;
                    state <= S_ROUND;
                end
                S_ROUND: begin
                    if (ovf) begin
                        result <= {s, 15'h7C00};
                        flags  <= 5'b00101;
                    end else begin
                        result <= {s, Er[4:0], frac};
                        flags  <= {3'b000, unf, inexact};
                    end
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
