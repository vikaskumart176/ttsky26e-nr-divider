/*
 * FP16 (IEEE 754 binary16) divider, Newton-Raphson reciprocal.
 * All four IEEE 754 rounding modes, full subnormal support, all five IEEE flags.
 * Bit-exact with model/fp16_div_model.py.
 *
 * Register-folded, minimum-precision datapath (49 cycles per divide; special
 * cases 1 cycle):
 *
 *   - Three working registers only: M (multiplicand), H (accumulator), L (scanned
 *     operand / low product bits). Every Newton-Raphson variable - reciprocal,
 *     residual, quotient, remainder - lives in the multiplier's own registers.
 *   - Residual-form iteration x <- x * (1 + e) with e = ~(b*x): the "2 - b*x"
 *     subtractor is a row of inverters, and x converges from below.
 *   - One-sided estimate: q1 = floor(a*x) + 1 is always Q or Q + 1, so the
 *     correction is a single conditional decrement.
 *   - Low-bits-only remainder: a*2^13 - q1*b lies in [-b, b), so only the low
 *     12 bits of q1*b decide the correction; a*2^13 contributes nothing to them.
 *     No remainder register or subtractor.
 *   - Radix-4 multiplier with digit 3 recoded as 4 - 1 (one carry flop), so no
 *     3*M adder.
 *   - Early-out: a power-of-two divisor (mantissa 1.0) needs no reciprocal; the
 *     quotient is the dividend mantissa and the divide takes 2 cycles.
 *   - Built-in self-check: the remainder multiply is preloaded with 2^11 - 1, so
 *     for a correct q1 the product is ma*2^13 + (2^11 - 1 - r) with r in [-mb, mb):
 *     its upper bits must equal 2*ma exactly and its low bits must lie within
 *     +/- mb. That holds only if q1 is Q or Q + 1, so an upset in the reciprocal
 *     or quotient multiplies that would give a wrong quotient raises "fault".
 *
 * Widths (15-bit reciprocal, 16-bit residual, 16-entry 6-bit seed) are the
 * minimum for which this is exact; proven over all mantissa pairs by
 * model/verify_mantissa.py.
 *
 * a and b must be held stable while busy is high (the wrapper guarantees it).
 */
`default_nettype none

module fp16_div_core (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire [1:0]  rmode,      // 0 nearest-even, 1 toward zero, 2 down (-inf), 3 up (+inf); sampled at start
    output reg  [15:0] result,
    output reg  [4:0]  flags,      // {invalid, divzero, overflow, underflow, inexact}
    output reg         fault,      // self-check failed on the last divide
    output wire        busy,
    output reg         done
);

    // Multiply phases. M is always the multiplicand.
    localparam [2:0] PH_BX0 = 3'd0,   // x0 * mb          -> residual e
                     PH_XT0 = 3'd1,   // x0 * (1 + e)     -> x1
                     PH_BX1 = 3'd2,   // x1 * mb          -> residual e
                     PH_XT1 = 3'd3,   // x1 * (1 + e)     -> x2
                     PH_QUO = 3'd4,   // x2 * ma (+1)     -> q1
                     PH_REM = 3'd5;   // q1 * mb + 2^11-1 -> remainder, self-check

    reg       run;
    reg [1:0] rm;                  // rounding mode of the divide in progress
    reg [2:0] ph;
    reg [3:0] cnt;                 // radix-4 steps left in this phase; 0 = phase boundary
    assign busy = run;

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
    wire [10:0] ma  = a_sub ? ({1'b0, fa} << lza) : {1'b1, fa};
    wire [10:0] mb  = b_sub ? ({1'b0, fb} << lzb) : {1'b1, fb};
    wire signed [7:0] Ea = a_sub ? (8'sd1 - $signed({4'd0, lza})) : $signed({3'd0, ea});
    wire signed [7:0] Eb = b_sub ? (8'sd1 - $signed({4'd0, lzb})) : $signed({3'd0, eb});
    wire signed [7:0] E0 = Ea - Eb + 8'sd15;

    wire invalid_n = (a_zero && b_zero) || (a_inf && b_inf) || a_snan || b_snan;
    wire nan_n     = a_nan || b_nan || invalid_n;
    wire divz_n    = b_zero && !a_zero && !a_inf && !a_nan;
    wire inf_n     = (a_inf && !b_inf && !b_nan) || divz_n;
    wire zero_n    = (a_zero && !b_zero && !b_nan && !a_nan) || (b_inf && !a_inf && !a_nan);
    wire special_n = nan_n || inf_n || zero_n;
    wire s         = sa ^ sb;
    wire pow2      = (mb[9:0] == 10'd0);                 // divisor mantissa is exactly 1.0

    // seed table: floor(64 / top of mantissa interval), never above 1/mb
    function [5:0] seed;
        input [3:0] i;
        begin
            case (i)
                4'd0:  seed = 6'd60; 4'd1:  seed = 6'd56; 4'd2:  seed = 6'd53; 4'd3:  seed = 6'd51;
                4'd4:  seed = 6'd48; 4'd5:  seed = 6'd46; 4'd6:  seed = 6'd44; 4'd7:  seed = 6'd42;
                4'd8:  seed = 6'd40; 4'd9:  seed = 6'd39; 4'd10: seed = 6'd37; 4'd11: seed = 6'd36;
                4'd12: seed = 6'd35; 4'd13: seed = 6'd34; 4'd14: seed = 6'd33; default: seed = 6'd32;
            endcase
        end
    endfunction

    // ---------------- datapath: M, H, L and the recode carry ----------------
    // {H, L} shifts right two bits per step. After n steps the product is
    // {H, L[17:18-2n]}; six-step phases leave it in {H, L[17:6]}, nine-step
    // phases in {H, L}.
    reg        [15:0] M;      // x (15 bits) or 4*q1
    reg signed [16:0] H;
    reg        [17:0] L;
    reg               c;

    // radix-4 digit with carry: 0, +M, +2M, -M (carry out), 0 (carry out)
    wire [2:0]  dig   = {2'b00, c} + {1'b0, L[1:0]};
    wire        d_sub = (dig == 3'd3);
    wire        d_one = (dig == 3'd1) | d_sub;
    wire        d_two = (dig == 3'd2);
    wire [18:0] pp_m  = d_two ? {2'b00, M, 1'b0} : d_one ? {3'b000, M} : 19'd0;
    wire [18:0] pp    = pp_m ^ {19{d_sub}};
    wire [18:0] sum   = {{2{H[16]}}, H} + pp + {18'd0, d_sub};

    // values handed from one phase to the next
    wire [15:0] resid = ~{H[12:0], L[17:15]};            // 1 - b*x, bits 2^-1 .. 2^-16
    wire [13:0] xq_hi = H[13:0];                         // new x[14:2], or q1
    wire        to_xt = (ph == PH_BX0) | (ph == PH_BX1); // next phase scans 1 + e
    wire        q_now = (ph == PH_QUO);                  // this phase produced q1

    // L[17:12] is only scanned in the nine-step phases, so it always loads {01, e[15:12]}.
    wire [11:0] l_lo  = !run            ? {1'b0, mb} :
                        to_xt           ? resid[11:0] :
                        (ph == PH_XT1)  ? {1'b0, ma} : {1'b0, mb};

    always @(posedge clk) begin
        if (!run || cnt == 4'd0) begin                   // phase boundary: load the next multiply
            L <= {2'b01, resid[15:12], l_lo};
            // PH_QUO starts at +1 at the quotient LSB; PH_REM at 4 * (2^11 - 1)
            H <= !run         ? 17'd0 :
                 (ph == PH_XT1) ? 17'h01000 :
                 (ph == PH_QUO) ? 17'h01FFC : 17'd0;
            c <= 1'b0;
            if (!run)
                M <= pow2 ? {ma, 5'd0} : {1'b0, seed(mb[9:6]), 9'd0};   // q = ma exactly, or x0
            else if (!to_xt)
                M <= {xq_hi, q_now ? 2'b00 : L[17:16]};
        end else begin
            H <= sum[18:2];
            L <= {sum[1:0], L[17:2]};
            c <= dig[2] | d_sub;
        end
    end

    // ---------------- remainder sign, self-check, correction ----------------
    // Valid at the end of PH_REM: {H, L[17:6]} = 4 * t, t = q1*mb + 2^11 - 1
    //   = ma*2^13 + (2^11 - 1 - r),  r = ma*2^13 - q1*mb.
    // For a correct q1, r is in [-mb, mb): t[11] is the sign of r, t[10:0] holds
    // ~r (r >= 0) or -r - 1 (r < 0), and the bits above them are exactly 2*ma.
    wire [11:0] rlo    = {H[1:0], L[17:8]};              // t mod 4096
    wire        r_sgn  = rlo[11];                        // remainder < 0: q1 = Q + 1
    wire [10:0] r_mag  = rlo[10:0] ^ {11{~r_sgn}};       // r, or -r - 1
    wire [11:0] r_cmp  = {1'b0, mb} + {1'b0, ~r_mag};    // mb - r_mag - 1; carry out = (r_mag < mb)
    wire        r_edge = ~|r_cmp[10:0];                  // r_mag == mb - 1, i.e. r == -mb when negative
    wire        r_neg  = r_sgn & ~pow2;                  // early-out: remainder is zero
    wire        rem_nz = ~pow2 & (r_sgn ? ~r_edge : (|r_mag));
    wire        chk_ok = pow2 | (r_cmp[11] & (H[16:2] == {3'd0, ma, 1'b0}));
    wire [13:0] q      = M[15:2] - {13'd0, r_neg};

    wire               nrm     = ~q[13];
    wire        [13:0] qn      = nrm ? {q[12:0], 1'b0} : q;
    wire signed [7:0]  En      = E0 - $signed({7'd0, nrm});
    wire        [10:0] keep_u  = qn[13:3];
    wire               rne     = (rm == 2'd0);
    wire               dir_up  = ((rm == 2'd2) & s) | ((rm == 2'd3) & ~s);   // directed mode rounds magnitude up
    wire               inc_u   = rne ? qn[2] & ((|qn[1:0]) | rem_nz | keep_u[0])
                                     : dir_up & (qn[2] | (|qn[1:0]) | rem_nz);
    wire               carry_u = inc_u & (&keep_u);
    wire               tiny    = (En < 0) || ((En == 0) && !carry_u);

    wire               subn    = (En <= 0);
    wire signed [7:0]  sh_s    = 8'sd1 - En;
    wire        [3:0]  sh      = !subn ? 4'd0 : (sh_s > 8'sd15) ? 4'd15 : sh_s[3:0];
    wire        [13:0] qd      = qn >> sh;
    wire        [13:0] lost    = qn & ~({14{1'b1}} << sh);
    wire        [10:0] keep    = qd[13:3];
    wire               rnd     = qd[2];
    wire               stk     = (|qd[1:0]) | rem_nz | (|lost);
    wire               inexact = rnd | stk;
    wire               inc     = rne ? rnd & (stk | keep[0]) : dir_up & inexact;
    wire        [11:0] keep2   = {1'b0, keep} + {11'd0, inc};
    wire signed [7:0]  Eb0     = subn ? 8'sd0 : En;
    wire signed [7:0]  Er      = keep2[11] ? (Eb0 + 8'sd1) :
                                 (subn && keep2[10]) ? 8'sd1 : Eb0;
    wire        [9:0]  frac    = keep2[11] ? keep2[10:1] : keep2[9:0];
    wire               ovf     = (Er >= 8'sd31);
    wire               unf     = tiny & inexact;

    // ---------------- control ----------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run    <= 1'b0;
            rm     <= 2'd0;
            ph     <= 3'd0;
            cnt    <= 4'd0;
            result <= 16'd0;
            flags  <= 5'd0;
            fault  <= 1'b0;
            done   <= 1'b0;
        end else begin
            done <= 1'b0;
            if (!run) begin
                if (start) begin
                    fault <= 1'b0;
                    if (special_n) begin
                        result <= nan_n ? 16'h7E00 :
                                  inf_n ? {s, 15'h7C00} : {s, 15'd0};
                        flags  <= {invalid_n, divz_n, 3'b000};
                        done   <= 1'b1;
                    end else begin
                        run <= 1'b1;
                        rm  <= rmode;
                        ph  <= pow2 ? PH_REM : PH_BX0;    // early-out goes straight to rounding
                        cnt <= pow2 ? 4'd0 : 4'd6;
                    end
                end
            end else if (cnt != 4'd0) begin
                cnt <= cnt - 4'd1;
            end else if (ph == PH_REM) begin
                if (ovf) begin
                    result <= (rne | dir_up) ? {s, 15'h7C00} : {s, 15'h7BFF};   // inf or largest finite
                    flags  <= 5'b00101;
                end else begin
                    result <= {s, Er[4:0], frac};
                    flags  <= {3'b000, unf, inexact};
                end
                fault <= ~chk_ok;
                done <= 1'b1;
                run  <= 1'b0;
            end else begin
                ph  <= ph + 3'd1;
                cnt <= to_xt ? 4'd9 : 4'd6;
            end
        end
    end

endmodule
