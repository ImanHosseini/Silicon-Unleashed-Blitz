`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/23/2021 06:03:33 PM
// Design Name: 
// Module Name: convo
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module convo(
    input [7:0] px1,
    input [7:0] px2,
    input [7:0] px3,
    input [7:0] px4,
    input [7:0] px5,
    input [7:0] px6,
    input [7:0] px7,
    input [7:0] px8,
    input [7:0] px9,
    input [7:0] fk1,
    input [7:0] fk2,
    input [7:0] fk3,
    input [7:0] fk4,
    input [7:0] fk5,
    input [7:0] fk6,
    input [7:0] fk7,
    input [7:0] fk8,
    input [7:0] fk9,
    input clk,
    output reg [15:0] data_out,
    output reg data_rdy = 1'd0
    );

reg [3:0] state = 4'd0;
reg [7:0] px;
reg [7:0] fk;
reg [3:0] sel = 4'd0;
wire carry;
reg [15:0] csum;
reg [15:0] sum;

mux_9to1 pmux(px1,px2,px3,px4,px5,px6,px7,px8,px9,sel,px);
mux_9to1 kmux(fk1,fk2,fk3,fk4,fk5,fk6,fk7,fk8,fk9,sel,fk);
mult_8 mult(.CLK(clk),.A(px),.B(fk),.P(mult_o));
c_add_16 adder(.A(mult_o),.B(sum),.CLK(clk),.C_OUT(carry),.S(csum));


assign data_out = sum;

assign sum = (carry==1)? 16'hFFFF : csum;
always @ (posedge clk) begin
//sum <= (carry==1)? 16'hFFFF : csum;
case (state) 
1,2,3,4,5,6,7,8,9,10: begin
state <= (state+1);
sel <= sel+1;
end
0: begin 
sum <= 0;
state <= (state+1);
data_rdy <= 0;
end
11:begin
 state <= 0;
 data_rdy <= 1;
end

endcase
end
endmodule
