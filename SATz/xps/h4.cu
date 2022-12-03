#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>

/*
    only god (and my adviser) can judge me, sorry Scott Meyers!
    & bible doesnt say anything about not scoping into std.
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

#define V 59790
#define TIMES 5

// An experiment with cuRAND

template <int S>
__global__ void f(bool *arr, int vnum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

int main()
{
    for (auto i = 0; i < TIMES; i++)
    {
        auto t0 = chrono::steady_clock::now();
        
        auto t1 = chrono::steady_clock::now();
        chrono::duration<double, micro> elapsed = t1 - t0;
        printf("[%s] %lf (us)\n", TNAME, elapsed.count());
    }
}
