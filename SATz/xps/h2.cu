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
#define VN 59790
#endif

#ifndef UR
#define UR 32
#endif

#define TIMES 5

// An experiment with cuRAND

// f0: base version
__global__ void f0(bool *arr, int tcnt, uint64_t seed, int s)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tcnt) return;
    curandState *randS = (curandState *)malloc(sizeof(curandState));
    curand_init(idx*seed, 0, 0, randS);
    for(int i=0; i<s; i++){
        bool rnd = curand(randS)%2;
        arr[idx*s+ i] = rnd;
    }
    free(randS);
}



// f0 + Loop Unrolling [S=64 won here]
template <int S>
__global__ void f1(bool *arr, int tcnt, uint64_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tcnt) return;
    curandState *randS = (curandState *)malloc(sizeof(curandState));
    curand_init(idx*seed, 0, 0, randS);
    #pragma unroll (S)
    for(int i=0; i<S; i++){
        bool rnd = curand(randS)%2;
        arr[idx*S+ i] = rnd;
    }
    free(randS);
}

int main()
{
    for (auto i = 0; i < TIMES; i++)
    {
        printf("[UR: %d | VN: %d]\n",UR, VN);
        int Vcnt = (VN+UR-1)/UR;
        int bsz = (511+Vcnt)/512;
        bool* d_arr;
        // padded to get rid of remainder loop
        cudaMalloc(&d_arr, sizeof(bool)*VN);
        // look, i know memory calls sync and this is not needed.
        cudaDeviceSynchronize();
        auto t0 = chrono::steady_clock::now();
        // f1<UR><<<bsz,512>>>(d_arr,Vcnt,(uint64_t)time(0));
        f0<<<bsz,512>>>(d_arr,Vcnt,(uint64_t)time(0),UR);
        cudaDeviceSynchronize();
        auto t1 = chrono::steady_clock::now();
        chrono::duration<double, micro> elapsed = t1 - t0;
        printf("[%s] %.17g (us)\n", "time", elapsed.count());
    }
}
