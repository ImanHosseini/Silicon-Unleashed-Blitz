#include <cstdlib>
#include <chrono>
#include <cstdio>

using namespace std;

#define N 59790

void randN(int *ptr, int n, int mx)
{
    for (int i = 0; i < N; i++)
    {
        ptr[i] = rand() % mx;
    }
}

template <int size>
__global__ void pullup(int *arr, int ix, int vnum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int ic = idx / size;
#pragma unroll(size)
        for (auto i = 0; i < size; i++)
        {
            int ii = idx*size + i;
            if (ii == ix) continue;
            if (ii >= vnum) break;
            arr[idx*size+i] = 1;
        }
}

__global__ void pullupX8(int *arr, int ix, int vnum, int ixX)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == ixX || (idx+1)*4>vnum){
        for(auto i = 0; i<4; i++){
            int ii = idx*4 + i;
            if( ii >= vnum) break;
            if( ii == ix) continue;
            arr[ii] = 1;
        }
    }else{
        arr[idx*4] = 1;
        arr[idx*4+1] = 1;
        arr[idx*4+2] = 1;
        arr[idx*4+3] = 1;
    }
}

#define S 8

int main()
{
    int* h_arr = (int*)malloc(sizeof(int)*N);
    // randN(h_arr,N,)
    int* d_arr;
    cudaMalloc(&d_arr,sizeof(int)*N);
    cudaDeviceSynchronize();
    int ths = (N+S-1)/S;
    int bsz = (ths+511)/512;
    int tsz = 512;
    auto t0 = chrono::steady_clock::now();
    pullup<S><<<bsz,tsz>>>(d_arr,13,N);
    cudaDeviceSynchronize();
    auto t1 = chrono::steady_clock::now();
    chrono::duration<double,std::micro> elapsed = t1 - t0;
    printf("%lf (us)\n",elapsed.count());
}
