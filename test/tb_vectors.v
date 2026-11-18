// Drives fp16_div_core with vector files (expected values from Berkeley SoftFloat,
// written by model/verify_softfloat.py) and compares result, flags and the
// self-check bit (which must stay 0). Plain Verilog, no cocotb:
//
//   python ../model/verify_softfloat.py
//   iverilog -o tb_vectors tb_vectors.v ../src/fp16_div_core.v
//   vvp tb_vectors +VEC=testfloat_l1.hex
//
// One 64-bit hex word per line: a (16) b (16) mode (8) result (16) flags (8).
`default_nettype none
`timescale 1ns / 1ps
module tb_vectors;
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
    integer fd, code, n, errors, cycles, max_cycles;
    initial begin
        if (!$value$plusargs("VEC=%s", path)) begin
            $display("usage: vvp tb_vectors +VEC=file.hex");
            $finish;
        end
        fd = $fopen(path, "r");
        if (fd == 0) begin
            $display("cannot open vector file");
            $finish;
        end
        n = 0; errors = 0; max_cycles = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        code = $fscanf(fd, "%h", vec);
        while (code == 1) begin
            @(negedge clk);
            a  = vec[63:48];
            b  = vec[47:32];
            rm = vec[25:24];
            start = 1;
            @(negedge clk);
            start = 0;
            rm = ~rm;                    // the core must have latched the mode at start
            cycles = 1;
            while (!done) begin @(negedge clk); cycles = cycles + 1; end
            if (cycles > max_cycles) max_cycles = cycles;
            if (result !== vec[23:8] || flags !== vec[4:0] || fault !== 1'b0) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("MISMATCH a=%h b=%h mode=%0d got %h/%b fault=%b exp %h/%b",
                             a, b, vec[25:24], result, flags, fault, vec[23:8], vec[4:0]);
            end
            n = n + 1;
            code = $fscanf(fd, "%h", vec);
        end
        $display("vectors=%0d errors=%0d max_cycles=%0d", n, errors, max_cycles);
        $finish;
    end
endmodule
