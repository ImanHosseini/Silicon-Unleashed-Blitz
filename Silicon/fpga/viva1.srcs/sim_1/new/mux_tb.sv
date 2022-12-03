module mux_tb();
// Declare inputs as regs and outputs as wires
reg [7:0] a1,a2,a3,a4,a5,a6,a7,a8,a9;
reg clock;
reg [3:0] sel = 0;
reg [7:0] out;
// Initialize all variables
initial begin        
  clock = 1;       // initial value of clock
  a1 = 8'd1;
  a2 = 8'd2;
  a3 = 8'd3;
  a4 = 8'd4; 
  a5 = 8'd5;
  a6 = 8'd6;                           
  a7 = 8'd7;
  a8 = 8'd8;
  a9 = 8'd9;

  #5 sel = 4'd1;    
  #10 sel = 4'd2;   
  #10 sel = 4'd3;  
  #10 sel = 4'd4; 
  #5 $finish;  
end

// Clock generator
always begin
  #5 clock = ~clock; // Toggle clock every 5 ticks
end

// Connect DUT to test bench
mux_9to1 mux(
a1,a2,a3,a4,a5,a6,a7,a8,a9,sel,out
);

endmodule