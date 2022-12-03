module convo_tb();
// Declare inputs as regs and outputs as wires
reg clock, reset, enable;
reg [7:0] px1 = 8'd1;
reg [7:0] px2 = 8'd2;
reg [7:0] px3 = 8'd3;
reg [7:0] px4 = 8'd4;
reg [7:0] px5 = 8'd5;
reg [7:0] px6 = 8'd6;
reg [7:0] px7 = 8'd7;
reg [7:0] px8 = 8'd8;
reg [7:0] px9 = 8'd9;
reg [7:0] fk1 = 8'd3;
reg [7:0] fk2 = 8'd1;
reg [7:0] fk3 = 8'd2;
reg [7:0] fk4 = 8'd0;
reg [7:0] fk5 = 8'd1;
reg [7:0] fk6 = 8'd2;
reg [7:0] fk7 = 8'd1;
reg [7:0] fk8 = 8'd1;
reg [7:0] fk9 = 8'd1;
reg [15:0] dout = 16'b0;
reg drdy = 1'b0;
reg [3:0] sel;
reg [15:0] mo;
reg [15:0] ao;
reg [7:0] ma;
reg [7:0] mb;

// Initialize all variables
initial begin        
  $display ("time\t clk reset enable counter");	
  $monitor ("%g\t %d %d", 
	  $time,dout,drdy);	
  clock = 1;       // initial value of clock
  reset = 0;       // initial value of reset
  enable = 0;      // initial value of enable
  #800 $finish;      // Terminate simulation
end

// Clock generator
always begin
  #5 clock = ~clock; // Toggle clock every 5 ticks
end

// Connect DUT to test bench
convo u_convo(
.px1(px1),
.px2(px2),
.px3(px3),
.px4(px4),
.px5(px5),
.px6(px6),
.px7(px7),
.px8(px8),
.px9(px9),
.fk1(fk1),
.fk2(fk2),
.fk3(fk3),
.fk4(fk4),
.fk5(fk5),
.fk6(fk6),
.fk7(fk7),
.fk8(fk8),
.fk9(fk9),
.clk(clock),
.data_out(dout),
.data_rdy(drdy),
.mlt_o(mo),
.sel_o(sel),
.a_o(ao),
.ma(ma), 
.mb(mb)
);

endmodule