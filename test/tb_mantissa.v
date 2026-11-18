// Drives fp16_div_core with all 1024 x 1024 normalised mantissa pairs and compares
// against vectors from model/gen_mantissa_vectors.py. Plain Verilog, no cocotb:
//   python ../model/gen_mantissa_vectors.py mantissa.hex
//   iverilog -o tb_mantissa tb_mantissa.v ../src/fp16_div_core.v && vvp tb_mantissa
`default_nettype none
`timescale 1ns / 1ps
module tb_mantissa;
    reg         clk = 0, rst_n = 0, start = 0;
    reg  [15:0] a = 0, b = 0;
    wire [15:0] result;
    wire [4:0]  flags;
    wire        fault, busy, done;
    fp16_div_core dut (.clk(clk), .rst_n(rst_n), .start(start), .a(a), .b(b), .rmode(2'd0),
                       .result(result), .flags(flags), .fault(fault), .busy(busy), .done(done));
    always #5 clk = ~clk;

    reg [20:0] expv [0:1048575];
    integer i, errors, cycles, max_cycles;
    initial begin
        $readmemh("mantissa.hex", expv);
        errors = 0; max_cycles = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        for (i = 0; i < 1048576; i = i + 1) begin
            @(negedge clk);
            a = 16'h3C00 | i[19:10];
            b = 16'h3C00 | i[9:0];
            start = 1;
            @(negedge clk);
            start = 0;
            cycles = 1;
            while (!done) begin @(negedge clk); cycles = cycles + 1; end
            if (cycles > max_cycles) max_cycles = cycles;
            if ({flags, result} !== expv[i] || fault !== 1'b0) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("MISMATCH a=%h b=%h got %h/%b fault=%b exp %h/%b", a, b, result, flags, fault,
                             expv[i][15:0], expv[i][20:16]);
            end
        end
        $display("pairs=%0d errors=%0d max_cycles=%0d", i, errors, max_cycles);
        $finish;
    end
endmodule
