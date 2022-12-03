`timescale 1 ns/1 ns
module top_tb();
reg clk;
reg sw = 0;
reg uart_out;
initial begin

end

/*
module top(
    input wire sw,
    input clk,
    input UART_TXD_IN,
    output logic [3:0] LED,
    output UART_RXD_OUT
    );
*/
top dut(.sw(sw),.clk(clk),.UART_TXD_IN(1),.LED(),.UART_RXD_OUT(uart_out));

always
begin
clk <=0; #10;
clk <=1; #10;
end
endmodule
