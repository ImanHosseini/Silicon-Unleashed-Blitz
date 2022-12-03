module sobel 
    #(parameter IMAGE_WIDTH = 8,
    parameter IMAGE_HEIGHT = 8,
    parameter KERNEL_WIDTH = 5,
    parameter KERNEL_HEIGHT = 5) (
    input [7:0] kernel[KERNEL_HEIGHT-1:0][KERNEL_WIDTH-1:0],
    input [7:0] img_in[IMAGE_HEIGHT-1:0][IMAGE_WIDTH-1:0],
    output wire [15:0] img_out[IMAGE_HEIGHT-1:0][IMAGE_WIDTH-1:0]
);

integer i;
integer j;
integer k;
integer l;
integer ix;
integer iy;     
reg [15:0] sum_comb[IMAGE_HEIGHT][IMAGE_WIDTH];
always @* begin 
  for( i=0; i< IMAGE_HEIGHT; i=i+1) begin
      for( j=0; j< IMAGE_WIDTH; j=j++) begin
          sum_comb[i][j] = 'd0;
          for ( k=0; k<KERNEL_HEIGHT; k++) begin
              for ( l=0;l<KERNEL_WIDTH;l++) begin
                
                ix = i - KERNEL_HEIGHT/2 + k;
                iy = j - KERNEL_WIDTH/2 + l;
                if (ix>=0 || iy>=0 || ix<IMAGE_HEIGHT || iy<IMAGE_WIDTH) begin
                        sum_comb[i][j] = sum_comb[i][j] + img_in[ix][iy]*kernel[k][l];
                end
            end
      end
  end
end
end
genvar ii,jj;
generate
  for( ii=0; ii< IMAGE_HEIGHT; ii=ii+1) begin
      for( jj=0; jj< IMAGE_WIDTH; jj=jj+1) begin
        assign img_out[ii][jj] = sum_comb[ii][jj];
      end 
  end
endgenerate
endmodule
// generate
// for (i = 0; i < IMAGE_WIDTH ; i++) {
//     for (j = 0; j<IMAGE_HEIGHT; j++){
//           assign img_out[i][j] = i; 
//     }    
// }
// endgenerate 