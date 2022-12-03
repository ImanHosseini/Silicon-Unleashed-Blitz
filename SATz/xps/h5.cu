#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>

/*
    only god (and my adviser) can judge me, sorry Scott Meyers!
    bible doesnt say anything about not scoping into std.
*/
using namespace std;

#define TC_S 1
#define TC_M 2

#ifndef TC
#define TC TC_S
#endif

#if TC == TC_S
#define TNAME "single"
#endif

#if TC == TC_M
#define TNAME "multi"
#endif

const char *tname = TNAME;

#ifndef VN
#define VN 5200*512
#endif

#ifndef UR
#define UR 16
#endif

#define TIMES 5

// An experiment with cuRAND II (To Stride Or Not?)
/*
    STRIDED IS SLIGHTLY BETTER FOR LARGE V (V ~ 8000)
    BUT IT IS __MUCH___ BETTER FOR SMALL V (V ~ 900)
    it's probably a cache window thing
    see: https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
    Im actually surprised that at V=900 strided is ~2x faster
    For V=8000 there is little difference no matter S
    Generally only stride if you have to (mult-dimensional array) and then keep stride elements small 
    ALSO unroll helps.. if you pad, which you don't have to if you dont unroll
    the padding makes unroll not worth it.
*/

template<int Q>
__global__ void csi(curandState_t* cst, uint64_t seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(idx+seed, 0, 0, cst+idx);
}

// Strided
template <int S>
__global__ void f2(uint8_t *arr, int V, curandState* cst)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x;
    // curandState *randS = (curandState *)malloc(sizeof(curandState));
    // curand_init(idx*seed, 0, 0, randS);
    for(int i=0; i<S; i++){
        int j = idx + i*stride;
        if (j>= V) break;
        arr[j] = curand(cst+idx);
    }
    // free(randS);
}

void tc(int& bsz, int& tsz, int v){
    if(v < 512*UR){
        bsz = 1;
        tsz = (v + UR -1)/UR;
    }else{
        bsz = (v + 512*UR - 1)/ (512*UR);
        tsz = 512;
        int r = v%512;
        if( r < 8){
            bsz -= 1;
            tsz += (r/bsz);
        }
    }
    printf("<<<%d,%d>>>\n",bsz,tsz);
}

int main()
{
 
    int vsize = (VN + 7) / 8;
    int bsz,tsz;
        tc(bsz,tsz,vsize);
    curandState_t* cst;
    cudaMalloc(&cst, sizeof(curandState_t)*bsz*tsz);
    csi<1><<<bsz,tsz>>>(cst,(uint64_t)time(0));
    for (auto i = 0; i < TIMES; i++)
    {
        printf("[UR: %d | VN: %d]\n",UR, VN);
        int bsz,tsz;
        tc(bsz,tsz,vsize);
        uint8_t* d_arr;
        // padded to get rid of remainder loop
        cudaMalloc(&d_arr, sizeof(uint8_t)*vsize);
        // look, i know memory calls sync and this is not needed.
        cudaDeviceSynchronize();
        auto t0 = chrono::steady_clock::now();
        f2<UR><<<bsz,tsz>>>(d_arr,vsize,cst);
        // f0<<<bsz,512>>>(d_arr,Vcnt,(uint64_t)time(0),UR);
        cudaDeviceSynchronize();
        auto t1 = chrono::steady_clock::now();
        chrono::duration<double, micro> elapsed = t1 - t0;
        printf("[%s] %.17g (us)\n", "time", elapsed.count());
    }
}
