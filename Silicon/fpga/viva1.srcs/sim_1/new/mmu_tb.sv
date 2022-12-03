module mmu_tb();

reg clk;
wire [18:0] a1, a2,a3,a4,a5,a6,a7,a8,a9;
wire [7:0] r1,r2,r3,r4,r5,r6,r7,r8,r9;
wire [7:0] p1,p2,p3,p4,p5,p6,p7,p8,p9;

initial begin        
  clk = 1;       // initial value of clock
  #80000 $finish;      // Terminate simulation
end

blk_mem_gen_0 ram1(.clka(clk),.ena(1),.wea(0),.addra(a1),.douta(r1));
blk_mem_gen_0 ram2(.clka(clk),.ena(1),.wea(0),.addra(a2),.douta(r2));
blk_mem_gen_0 ram3(.clka(clk),.ena(1),.wea(0),.addra(a3),.douta(r3));
blk_mem_gen_0 ram4(.clka(clk),.ena(1),.wea(0),.addra(a4),.douta(r4));
blk_mem_gen_0 ram5(.clka(clk),.ena(1),.wea(0),.addra(a5),.douta(r5));
blk_mem_gen_0 ram6(.clka(clk),.ena(1),.wea(0),.addra(a6),.douta(r6));
blk_mem_gen_0 ram7(.clka(clk),.ena(1),.wea(0),.addra(a7),.douta(r7));
blk_mem_gen_0 ram8(.clka(clk),.ena(1),.wea(0),.addra(a8),.douta(r8));
blk_mem_gen_0 ram9(.clka(clk),.ena(1),.wea(0),.addra(a9),.douta(r9));
px_mmu mmu(
.addr1(a1),
.addr2(a2),
.addr3(a3),
.addr4(a4),
.addr5(a5),
.addr6(a6),
.addr7(a7),
.addr8(a8),
.addr9(a9),
.port1(r1),
.port2(r2),
.port3(r3),
.port4(r4),
.port5(r5),
.port6(r6),
.port7(r7),
.port8(r8),
.port9(r9),
.clk(clk),
.done(done),
.px1(p1),
.px2(p2),
.px3(p3),
.px4(p4),
.px5(p5),   
.px6(p6),
.px7(p7),
.px8(p8),
.px9(p9)
);

always begin
  #5 clk = ~clk; // Toggle clock every 5 ticks
end

endmodule
