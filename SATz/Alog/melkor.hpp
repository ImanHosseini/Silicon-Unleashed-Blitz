#ifndef _melkor_hpp_INCLUDED
#define _melkor_hpp_INCLUDED

#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <csignal>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdint>

#define BFACTOR 18

#define DEBUG
#define WITNESS_GEN

#if defined(DEBUG)
#define ccE(err) __checkCudaErrors(err, __FILE__, __LINE__)
#else
#define ccE(err) err
#endif

#if defined(DEBUG)
#define dprint(...) printf(__VA_ARGS__)
#else
#define dprint(...)
#endif

namespace Melkor
{
         inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
    {
        const char *ename;
        const char *estr;
        ename = cudaGetErrorName(err);
        estr = cudaGetErrorString(err);
        if (cudaSuccess != err)
        {
            fprintf(stderr,
                    "CUDA Runtime API error = %04d from file <%s>, line %i:\n",
                    err, file, line);
            fprintf(stderr, "\t%s : %s\n", ename, estr);
            exit(-1);
        }
    }


    

}

#include "parser.hpp"
// #include "common.cuh"
#include "sls.cuh"

#define ERR(x)                                                       \
    do                                                               \
    {                                                                \
        fprintf(stderr, "[ERR]:%s:%d %s\n", __FILE__, __LINE__, #x); \
        exit(-1);                                                    \
    } while (0)

#endif