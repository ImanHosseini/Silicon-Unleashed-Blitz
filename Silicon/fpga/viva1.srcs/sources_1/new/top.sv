`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/10/2020 07:03:48 AM
// Design Name: 
// Module Name: top
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


module top(
    input wire sw,
    input clk,
    input UART_TXD_IN,
    output logic [3:0] LED,
    output UART_RXD_OUT
    );
    reg wr_en = 0;
    reg [7:0] din;
    wire busy;
    wire [1:0] sout;
    reg rdy_clr;
    reg [7:0] dout;
    reg [7:0] my_str [0:5];
    reg [7:0] c_data = 8'hff;
    assign LED[1:0] = sout;
    assign LED[2] = busy;
    reg [17:0] cidx = 0;
    
    parameter [2:0] START = 3'b000;
    reg [2:0] ustate = START;
    parameter [2:0] SEND_CH= 3'b001;
    parameter [2:0] RDY_L = 3'b010;
    parameter [2:0] WAIT_L = 3'b011;
    parameter [2:0] DEAD = 3'b100;
    parameter [2:0] RSTART = 3'b101;
    parameter [2:0] CONV = 3'b110;
    integer t = 0;
    integer x = 0;
    integer y = 0;
    integer w;
    integer h;
    integer isize;
//    reg [18:0] a1a,a1b,a2a,a2b,a3a,a3b;
    reg [18:0] addr = 0;
    wire [7:0] data; 
    reg [7:0] p1 = 0,p2 = 0,p3 = 0,p4 = 0,p5 = 0,p6 = 0,p7 = 0,p8 = 0,p9 = 0;
    wire [7:0] cd_out;
    reg [7:0] o_out;
    reg [17:0] o_addr = 0;
    reg o_we = 0;
    integer r0 = 0;
    integer md = 0;
    wire txclk_en;
//    uart uart0(.clk_50m(clk),.wr_en(wr_en),.din(din),.tx(UART_RXD_OUT),.rx(UART_TXD_IN),.rdy(rdy),.tx_busy(tx_busy),.rdy_clr(rdy_clr),.dout(dout));
    baud_rate_gen uart_baud(.clk_50m(clk),
			.rxclk_en(),
			.txclk_en(txclk_en));
    transmitter utx0(.clk_50m(clk),.clken(txclk_en),.din(c_data),.wr_en(wr_en),.tx_busy(busy),.tx(UART_RXD_OUT),.state_led(sout));    
    always @(posedge clk) begin // BAA
    case (ustate)
    START: begin // BSTART
    case (r0)
    0: begin
    r0 <= 1;
    end
    1: begin
    w <= 4*data;
    r0 <= 2;
    addr <= 19'd1;
    end
    2: begin 
    if(md==0) md <= 1;
    else begin
    h <= 4*data;
    isize <= w*data*4;
    ustate <= CONV;
    r0 <= 0;
    addr <= 19'd0;
    o_addr <= 0;
    md <= 0;
    end
    end
    endcase
    end
    CONV: begin
    case (r0)
    0: begin
    if(md==0) md <= 1;
    else begin
    if ((x-1)<0 || (y-1)<0) begin
    p1 <= 8'd0;
    end else begin
    p1 <= data;
    end
    addr <= 2 + (y-1)*w + x;
    r0 <= 1;
    md <= 0;
    end
    end
    1: begin
    if (md==0) md <= 1;
    else begin
    if ((y-1)<0) begin
    p2 <= 8'd0;
    end else begin
    p2 <= data;    
    end
    addr <= 2 + (y-1)*w + x+1;
    r0 <= 2;
    md <= 0;
    end
    end
    2: begin
    if (md == 0) md <= 1;
    else begin
    if ((x+1)>=w || (y-1)<0) begin
    p3 <= 8'd0;
    end else begin
    p3 <= data;
    end
    addr <= 2 + y*w + x-1;
    r0 <= 3;
    md <= 0;
    end
    end
    3: begin
    if (md==0) md <= 1;
    else begin
    if ((x-1)<0) begin
    p4 <= 8'd0;
    end else begin
    p4 <= data;
    end
    addr <= 2 + y*w + x;
    r0 <= 4;
    md <= 0;
    end
    end
    4: begin
    if(md==0) md <= 1;
    else begin
    p5 <= data;
    addr <= 2 + y*w + x+1;
    r0 <= 5;
    md <= 1;
    end
    end
    5: begin
    if (md==0) md <= 1;
    else begin
    if((x+1)>=w) begin
    p6 <= 8'd0;
    end else begin
    p6 <= data;
    end
    addr <= 2 + (y+1)*w + x-1;
    r0 <= 6;
    md <= 0;
    end
    end
    6: begin
    if (md == 0) md<=1;
    else begin
    if((y+1)>=h || (x-1)<0) begin
    p7 <= 8'd0;
    end else begin
    p7 <= data;
    end
    addr <= 2 + (y+1)*w + x;
    r0 <= 7;
    md <=0;
    end
    end
    7: begin
    if (md==0) md<=1;
    else begin
    if ((y+1)>=h) begin
    p8 <= 8'd0;
    end else begin
    p8 <= data;
    end
    addr <= 2 + (y+1)*w + x +1;
    r0 <= 8;
    md <=0;
    end
    end 
    8: begin
    if (md == 0) md <= 1;
    else begin
    if ((y+1)>=h || (x+1)>=w) begin
    p9 <= 8'd0;
    end else begin
    p9 <= data;
    end
    r0 <= 9; 
    o_we <= 1;
    md <= 0;
    end
    end
    9: begin
    o_we <= 0;
    if ((x==(w-1)) && (y==(h-1))) begin
    md <= 0;
    ustate <= RSTART;
    o_addr <= 0;
    end else begin
    o_addr <= o_addr + 1;
    r0 <= 0;
    if((x+1)<w) begin
    x <= x+1;
    end else begin  
    x <= 0;
    y <= y+1;
    end
    end    
    end
    endcase
    end
    RSTART: begin
    if (md==0) md <=1;
    else begin
    ustate <= SEND_CH;
    c_data <= o_out; 
    md <= 0;
    end
    end
    SEND_CH: begin
    wr_en <= 1;
    ustate <= RDY_L;
    end
    RDY_L: begin
    if(busy) begin
    wr_en <= 0;
    ustate <= WAIT_L;
    end
    end
    WAIT_L: begin
    if (busy == 0 && (cidx+1)<isize) begin
        ustate <= RSTART;
        cidx <= cidx + 1;
        o_addr <= o_addr+1;
    end    
    end
    endcase
    end // BAA
//integer xf;
//always_comb begin 
//xf = (x==0)?x:x-1;
//a1a = 2 + xf + w*(y-1);
//a2a = 2 + xf + w*y;
//a3a = 2 + xf + w*(y+1);
//end
//always_comb begin
//a1b = 2 + w*(x+1) + y-1;
//a2b = 2 + w*(x+1) + y;
//a3b = 2 + w*(x+1) + y+1;
//end
//always_comb begin
//if ((x-1)<0 || (y-1)<0) begin
//p1 = 8'd0;
//end else begin
//p1 = r1a[7:0];
//end
//end
//always_comb begin
//if ((y-1)<0) begin
//p2 = 8'd0;
//end else begin
//if (x==0) begin
//p2 = r1a[7:0];
//end else begin
//p2 = r1a[15:8];
//end
//end
//end
//always_comb begin
//if ((x+1)>=w || (y-1)<0) begin
//p3 = 8'd0;
//end else begin
//p3 = r1b;
//end
//end
//always_comb begin
//if ((x-1)<0) begin
//p4 = 8'd0;
//end else begin
//p4 = r2a[7:0];
//end
//end
//always_comb begin
//if(x==0) begin
//p5 = r2a[7:0];
//end else begin
//p5 = r2a[15:8];
//end
//end
//always_comb begin
//if ((x+1)>=w) begin
//p6 = 8'd0;
//end else begin
//p6 = r2b;
//end
//end
//always_comb begin
//if ((x-1)<0 || (y+1)>=h) begin
//p7 = 8'd0;
//end else begin
//p7 = r3a[7:0];
//end
//end
//always_comb begin
//if ((y+1)>=h) begin
//p8 = 8'd0;
//end else begin
//if (x==0) begin
//p8 = r3a[7:0];
//end else begin
//p8 = r3a[15:8]; 
//end
//end
//end
//always_comb begin
//if ((x+1)>w || (y+1)>=h) begin
//p9 = 8'd0;
//end else begin
//p9 = r3b;
//end
//end


blk_mem_gen_1 ram_out(.clka(clk),.wea(o_we),.dina(cd_out),.douta(o_out),.addra(o_addr));
blk_mem_gen_0 ram1(.clka(clk),.wea(0),.addra(addr),.douta(data),.dina());


dotter conv(
.clk(clk),
.a1(p1),
.a2(p2),
.a3(p3),
.a4(p4),
.a5(p5),
.a6(p6),
.a7(p7),
.a8(p8),
.a9(p9),
.r(cd_out)
);

endmodule
