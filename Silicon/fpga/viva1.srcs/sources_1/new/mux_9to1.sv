`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/09/2021 09:00:45 AM
// Design Name: 
// Module Name: mux_9to1
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
module mux_9to1
    (
    input [7:0] a1,
    input [7:0] a2,
    input [7:0] a3,
    input [7:0] a4,
    input [7:0] a5,
    input [7:0] a6,
    input [7:0] a7,
    input [7:0] a8,
    input [7:0] a9,
    input [3:0] sel,
    output reg [7:0] out
    );
  function [7:0] select;
  input [7:0] a1,a2,a3,a4,a5,a6,a7,a8,a9;
  input [3:0] sel;
  case (sel)
    4'd0: select = a1;
    4'd1: select = a2;
    4'd2: select = a3;
    4'd3: select = a4;
    4'd4: select = a5;
    4'd5: select = a6;
    4'd6: select = a7;
    4'd7: select = a8;
    4'd8: select = a9;
    default: select = a1;
   endcase
  endfunction
assign out = select(a1,a2,a3,a4,a5,a6,a7,a8,a9,sel);
endmodule
