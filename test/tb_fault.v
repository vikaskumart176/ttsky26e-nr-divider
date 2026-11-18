// Fault-injection campaign for the built-in self-check.
//
// For each vector one random bit of the working registers (M, H, L or the recode
// carry) is inverted at one random clock of the divide - a single-event upset.
// Each run is classified as
//   masked   : result and flags still correct, fault = 0
//   detected : fault = 1
//   silent   : result or flags wrong, fault = 0
// and silent runs are broken down by the multiply phase the upset landed in.
//
//   python ../model/verify_softfloat.py
//   iverilog -o tb_fault tb_fault.v ../src/fp16_div_core.v
//   vvp tb_fault +VEC=finite.hex
`default_nettype none
`timescale 1ns / 1ps
module tb_fault;
    reg         clk = 0, rst_n = 0, start = 0;
    reg  [15:0] a = 0, b = 0;
    reg  [1:0]  rm = 0;
    wire [15:0] result;
    wire [4:0]  flags;
    wire        fault, busy, done;
    fp16_div_core dut (.clk(clk), .rst_n(rst_n), .start(start), .a(a), .b(b), .rmode(rm),
                       .result(result), .flags(flags), .fault(fault), .busy(busy), .done(done));
    always #5 clk = ~clk;

    reg [63:0]   vec;
    reg [1023:0] path;
    integer fd, code, n, skipped, masked, detected, silent, seed;
    integer at, bitsel, cyc, hit_ph, injected, i;
    integer silent_ph [0:5];
    integer total_ph  [0:5];
    wire wrong = (result !== vec[23:8]) || (flags !== vec[4:0]);

    initial begin
        if (!$value$plusargs("VEC=%s", path)) begin
            $display("usage: vvp tb_fault +VEC=file.hex");
            $finish;
        end
        fd = $fopen(path, "r");
        seed = 12345;
        n = 0; skipped = 0; masked = 0; detected = 0; silent = 0;
        for (i = 0; i < 6; i = i + 1) begin silent_ph[i] = 0; total_ph[i] = 0; end
        repeat (3) @(posedge clk);
        rst_n = 1;
        code = $fscanf(fd, "%h", vec);
        while (code == 1) begin
            at     = 1 + ({$random(seed)} % 48);     // clock of the divide to hit
            bitsel = {$random(seed)} % 52;           // 0-15 M, 16-32 H, 33-50 L, 51 c
            @(negedge clk);
            a  = vec[63:48];
            b  = vec[47:32];
            rm = vec[25:24];
            start = 1;
            @(negedge clk);
            start = 0;
            cyc = 1; injected = 0; hit_ph = 0;
            while (!done) begin
                if (cyc == at && busy) begin
                    hit_ph = dut.ph;
                    if (bitsel < 16)      dut.M[bitsel]      = ~dut.M[bitsel];
                    else if (bitsel < 33) dut.H[bitsel - 16] = ~dut.H[bitsel - 16];
                    else if (bitsel < 51) dut.L[bitsel - 33] = ~dut.L[bitsel - 33];
                    else                  dut.c              = ~dut.c;
                    injected = 1;
                end
                @(negedge clk);
                cyc = cyc + 1;
            end
            if (!injected) skipped = skipped + 1;
            else begin
                n = n + 1;
                total_ph[hit_ph] = total_ph[hit_ph] + 1;
                if (fault) detected = detected + 1;
                else if (wrong) begin
                    silent = silent + 1;
                    silent_ph[hit_ph] = silent_ph[hit_ph] + 1;
                end else masked = masked + 1;
            end
            code = $fscanf(fd, "%h", vec);
        end
        $display("upsets=%0d (skipped %0d short divides)", n, skipped);
        $display("masked=%0d detected=%0d silent=%0d", masked, detected, silent);
        for (i = 0; i < 6; i = i + 1)
            $display("  phase %0d: upsets=%0d silent=%0d", i, total_ph[i], silent_ph[i]);
        $finish;
    end
endmodule
