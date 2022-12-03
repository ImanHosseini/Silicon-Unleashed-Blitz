
#include <cstdlib>
#include <chrono>
#include <cstdio>


__global__ void k(){
    int idx = (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x + threadIdx.x;
    int tidx =  blockDim.x * blockIdx.x + threadIdx.x;
    // printf("[%d]: BLK<%d,%d>|TH:<%d>\n",idx,blockIdx.x,blockIdx.y,threadIdx.x);
    printf("%d\n",tidx);
    // printf("TH:")
}

int main()
{
    k<<<2,16>>>();
    cudaDeviceSynchronize();
}