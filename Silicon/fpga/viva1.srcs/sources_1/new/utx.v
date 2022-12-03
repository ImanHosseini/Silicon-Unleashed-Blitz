`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/24/2020 03:30:12 PM
// Design Name: 
// Module Name: utx
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


module utx(
    input wire send,
    input wire [7:0] data,
    input wire clk,
    output wire rdy,
    output wire tx
    );
 
parameter RDY = 2'b00;
parameter LD = 2'b01;
parameter BITSEND = 2'b10;
parameter [13:0] TMR_MAX = 14'b10100010110000;
parameter BITMAX = 10;
    
reg [9:0] txData;
reg [1:0] txState = RDY;

reg [13:0] bitTmr = 0;
reg bitDone;
reg txBit = 1;
integer bitIdx;

task bit_counting;
    if (txState == RDY) begin
        bitIdx <= 0;
    end else if (txState == LD) begin
        bitIdx <= bitIdx + 1;
    end
endtask

task tx_proc;
    if (send == 1)
        txData <= {1'b1, data, 1'b0}; 
endtask

task tx_bit;
    if (txState == RDY)
        txBit <= 1;
    else if (txState == LD)
        txBit <= txData[bitIdx];
endtask

task bit_timing;
    if (txState == RDY) begin
        bitTmr <= 0;
    end 
    else begin
        if (bitDone == 1) begin
            bitTmr <= 0;
        end
        else begin
            bitTmr <= bitTmr + 1;
        end
    end
endtask

task sm;
case (txState)
    RDY: begin
        if (send)
            txState <= LD;
    end
    LD: txState <= BITSEND;
    BITSEND: 
        begin
            if (bitDone == 1)
                if (bitIdx == BITMAX)
                    txState <= RDY;
                else
                    txState <= LD;
        end
    default: txState <= RDY;
endcase
endtask

assign tx = txBit;
assign rdy = (txState==RDY); 

always @(posedge clk) begin
    bit_timing;
    bit_counting;
    tx_bit;
    tx_proc;
    sm;
    if (bitTmr == TMR_MAX)
        bitDone <= 1;
    else begin
        bitDone <= 0;
    end
end
  
endmodule
