module stest();
// Declare inputs as regs and outputs as wires
reg clock, reset, enable;
reg [7:0] kernel[4:0][4:0];
reg [7:0] img_in[7:0][7:0];
wire [15:0] img_out[7:0][7:0];

// Initialize all variables
initial begin        
  $display ("time\t clk reset enable counter");	
  $monitor ("%g\t %d", 
	  $time,img_out[0][0]);	
  clock = 1;       // initial value of clock
  reset = 0;       // initial value of reset
  enable = 0;      // initial value of enable
for(int i=0; i<8; i=i+1) begin
    for(int j=0;j<8;j=j+1) begin
        img_in[i][j] = 'd1;
    end
end
for(int i=0; i<5; i=i+1) begin
    for(int j=0;j<5;j=j+1) begin
        kernel[i][j] = i;
    end
end

  #5 reset = 1;    // Assert the reset
  #10 reset = 0;   // De-assert the reset
  #10 enable = 1;  // Assert enable
  #100 enable = 0; // De-assert enable
  #5 $finish;      // Terminate simulation
end

// Clock generator
always begin
  #5 clock = ~clock; // Toggle clock every 5 ticks
end

// Connect DUT to test bench
sobel u_sobel (
kernel,
img_in,
img_out
);

endmodule