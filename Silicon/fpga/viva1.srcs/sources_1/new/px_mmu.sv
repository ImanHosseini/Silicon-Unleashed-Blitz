`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/25/2021 01:18:29 AM
// Design Name: 
// Module Name: px_mmu
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


module px_mmu(
    input clk,
    input [7:0] port1,
    input [7:0] port2,
    input [7:0] port3,
    input [7:0] port4,
    input [7:0] port5,
    input [7:0] port6,
    input [7:0] port7,
    input [7:0] port8,
    input [7:0] port9,
    output [18:0] addr1,
    output [18:0] addr2,
    output [18:0] addr3 ,
    output [18:0] addr4 ,
    output [18:0] addr5 ,
    output [18:0] addr6 ,
    output [18:0] addr7 ,
    output [18:0] addr8 ,
    output [18:0] addr9 ,
    output [7:0] px1,
    output [7:0] px2,
    output [7:0] px3,
    output [7:0] px4,
    output [7:0] px5,
    output [7:0] px6,
    output [7:0] px7,
    output [7:0] px8,
    output [7:0] px9,
    output reg done = 0,
    output reg [17:0] total
    );
reg [7:0] state = 0;
integer w;
integer h;
integer xi;
integer yi;
integer t = 0;
integer t0 = 0;
integer z = 1;
reg [7:0] mask1 = 8'hff;
reg [7:0] mask2 = 8'b11111111;
reg [7:0] mask3 = 8'b11111111;
reg [7:0] mask4 = 8'b11111111;
reg [7:0] mask5 = 8'b11111111;
reg [7:0] mask6 = 8'b11111111;
reg [7:0] mask7 = 8'b11111111;
reg [7:0] mask8 = 8'b11111111;
reg [7:0] mask9 = 8'b11111111;

assign addr1 =  2 + (yi-1)*w + (xi-1) - z;
assign addr2 = 2 + (yi-1)*w + xi - z;
assign addr3 = 2 + (yi-1)*w + xi+1;
assign addr4 = 2 + yi*w + xi-1;
assign addr5 = 2 + yi*w + xi;
assign addr6 = 2 + yi*w + (xi+1);
assign addr7 = 2 + (yi+1)*w + xi-1;
assign addr8 = 2 + (yi+1)*w + xi;
assign addr9 = 2 + (yi+1)*w + xi+1;
assign px1 = mask1 & port1;
assign px2 = mask2 & port2;
assign px3 = mask3 & port3;
assign px4 = mask4 & port4;
assign px5 = mask5 & port5;
assign px6 = mask6 & port6;
assign px7 = mask7 & port7;
assign px8 = mask8 & port8;
assign px9 = mask9 & port9;

always @ (posedge clk) begin
case (state)
0: begin
w <= port1*4;
h <= port2*4;
xi <= 0;
yi <= 0;
t0 <= t0 + 1;
if (t0>4) begin
state <= state+1;
z <= 0;
end
end
1: begin
if ((xi-1)>=0 && (yi-1)>=0) begin
mask1 <= 8'hff;
end else begin
mask1 <= 0;
end
if ((yi-1)>=0) begin
mask2 <= 8'hff;
end else begin
mask2 <= 8'h0;
end
if ((yi-1)>=0) begin
mask3 <= 8'hff;
end else begin
mask3 <= 8'h0;
end
if ((xi-1)>=0) begin
mask4 <= 8'hff;
end else begin
mask4 <= 8'h0;
end
mask5 <= 8'hff;
if ((xi+1)<w) begin
mask6 <= 8'hff;
end else begin
mask6 <= 8'h0;
end
if ((xi-1)>=0 && (yi+1)<h) begin
mask7 <= 8'hff;
end else begin
mask7 <= 8'h0;
end
if ((yi+1)<h) begin
mask8 <= 8'hff;
end else begin
mask8 <= 8'h0;
end
if ((xi+1)<w && (yi+1)<h) begin
mask9 <= 8'hff;
end else begin
mask9 <= 8'h0;
end
state <= 2;
t <= 0;
end
2:begin
if(t<7) begin
t <= (t+1);
end else begin
t <= 0;
if (xi==(w-1) && yi == (h-1)) begin
state <= 3;
end else begin
state <= 1;
if(xi==(w-1)) begin
xi <= 0;
yi <= yi+1;
end else
xi <= xi + 1;
end 
end
end
default: begin
done <= 1;
total <= w*h; 
end
endcase
end

endmodule
