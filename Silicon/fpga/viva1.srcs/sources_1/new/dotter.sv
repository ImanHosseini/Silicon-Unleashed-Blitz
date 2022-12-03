`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/25/2021 05:28:27 AM
// Design Name: 
// Module Name: dotter
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


module dotter(input [7:0] a1, 
              input [7:0] a2, 
              input [7:0] a3, 
              input [7:0] a4, 
              input [7:0] a5,
              input [7:0] a6,
              input [7:0] a7,
              input [7:0] a8,
              input [7:0] a9,
              output reg [7:0] r,
              input clk);
 integer i;
 reg [11:0] sum_comb;
 always_comb begin 
                sum_comb = 'd0;
                sum_comb = sum_comb + a1;
                sum_comb = sum_comb + a2*2;
                sum_comb = sum_comb + a3;
                sum_comb = sum_comb + a4*2;
                sum_comb = sum_comb + a5*4;
                sum_comb = sum_comb + a6*2;
                sum_comb = sum_comb + a7;
                sum_comb = sum_comb + a8*2;
                sum_comb = sum_comb + a9;
end
              
always @(posedge clk) begin
r <= (sum_comb >>4);
end

endmodule
