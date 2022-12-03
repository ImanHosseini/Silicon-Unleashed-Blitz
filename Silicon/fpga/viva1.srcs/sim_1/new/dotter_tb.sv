module dotter_tb();
reg clk;
reg [7:0] a1,a2,a3,a4,b1,b2,b3,b4;
wire [7:0] r;

initial begin        
  clk = 1;       // initial value of clock
  a1 = 8'd1;
  a2 = 8'd2;
  a3 = 8'd3;
  a4 = 8'd4;
  b1 = 8'd4;
  b2 = 8'd3;
  b3 = 8'd2;
  b4 = 8'd1;
  #10;
  a1 = 8'd255;
  a2 = 8'd255;
  a3 = 8'd255;
  a4 = 8'd255;
  b1 = 8'd6;
  b2 = 8'd2;
  b3 = 8'd2;
  b4 = 8'd6;
  #10;
  a1 = 8'd1;
  b3 = 8'd0;
  #10
  a2 = 8'd1;
  b4 = 8'd0;
  #200 $finish;      // Terminate simulation
end

always begin
  #5 clk = ~clk; // Toggle clock every 5 ticks
end

dotter dut(.a1(a1),.a2(a2),.a3(a3),.a4(a4),.b1(b1),.b2(b2),.b3(b3),.b4(b4),.clk(clk),.r(r));

endmodule
