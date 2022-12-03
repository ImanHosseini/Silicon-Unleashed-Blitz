#include "melkor.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <chrono>
#include <cuda_profiler_api.h>
#include <thread>

#define Nq 1
#define Nb 4
#define Nt 512
#define Nk Nb *Nt
#define KS 64

#define NJb 16
#define NJt 32
#define NJk NJb *NJt
#define TIMEOUT 5000

namespace Melkor
{
    std::mutex g_lock;

    typedef struct Q_data
    {
        cudaStream_t s;
        uint8_t *valQ = NULL;
        uint8_t *valM = NULL;
        bool vQF = false;
        bool vMF = false;
        uint32_t hMax = 0;
        uint32_t hQ = 0;
        uint32_t *hN = NULL;
        uint32_t *flg = NULL;
        curandState_t *cst;
        // uint8_t* bVS = NULL;
        bool doCpy = false;
        cub::KeyValuePair<int, int> *d_argmin = NULL;
        // Q_data() : s(NULL), valQ(NULL), valM(NULL), hMax(0), hQ(0), hN(NULL), bVS(NULL), d_argmin(NULL){}
    } Q_data;

    Q_data Qs[QS];
    // Q_data QB;

    // void launch_spec(int &bsz, int &tsz, int n)
    // {
    //     bsz = (n + 511) / 512;
    //     tsz = 512;
    // }

    // These are 'extern'ed in .cuh
    uint32_t h_vn;
    uint32_t h_cn;
    uint32_t h_ln;
    uint32_t h_rs;
    size_t vsize;
    int *indxs;
    int *lits;
    int *d_lits;
    int *d_indxs;

    __constant__ int c_ln;
    // wouldn it help to keep first 64KB of literals in constant memory?
    __constant__ uint32_t c_vn;
    __constant__ uint32_t c_cn;

    /*---- Time Keeping ------------------------------------*/
    std::chrono::time_point<std::chrono::steady_clock> t0;

    uint32_t bszL;
    uint32_t tszL;
    uint32_t bszC;
    uint32_t tszC;
    uint32_t bszI;
    uint32_t tszI;
    // uint32_t bszV;
    // uint32_t tszV;
    // __constant__ uint32_t bsz;
    // __constant__ uint32_t tsz;
    __constant__ int *c_lits;
    // __constant__ int *d_indxs_C;
    int *d_L2C;
    __constant__ int *c_L2C;

    // void Q2M( cudaStream_t s = sA){
    //     ccE(cudaMemcpyAsync(valMAX,valQ,sizeof(bool)*h_vn,cudaMemcpyDeviceToDevice,s));
    // }

    // utility func for debug printing gpu memory
    // BEWARE: device printf cannot handle too much data!
    template <typename T>
    __global__ void pN(const T *p, int n, char *tag = "M")
    {
        printf("%s:[", tag);
        for (int i = 0; i < n; i++)
        {
            printf("%d ", p[i]);
        }
        printf("]\n");
    }

    inline void kl_init(uint32_t &bsz, uint32_t &tsz, size_t v)
    {
        if (v < 512)
        {
            bsz = 1;
            tsz = v;
        }
        else
        {
            bsz = (v + 511) / 512;
            tsz = 512;
        }
        dprint("initVQ<<<%d,%d>>>\n", bsz, tsz);
    }

    inline void kl_LKX(uint32_t &bsz, uint32_t &tsz)
    {
        bsz = (h_ln + 512 * SX - 1) / 512 * SX;
        tsz = 512;
        dprint("LKX<<<%d,%d>>>\n", bsz, tsz);
    }

    // __global__ void BVSD(int clause, int c_vn)
    // {
    //     printf("c: %d | v: %d | %d\n", clause, c_vn, bVSC[c_vn * c_cn + clause]);
    // }

    // void printVH()
    // {
    //     bool* vals = new bool[h_vn];
    //     cudaMemcpy(vals,valMAX,sizeof(bool)*h_vn,cudaMemcpyDeviceToHost);
    //     cudaDeviceSynchronize();
    //     printf("V:");
    //     for(int i = 0; i<h_vn; i++){
    //         printf("%d",vals[i]);
    //     }
    //     printf("\n");
    // }

    //     void terminate()
    //     {
    //         std::chrono::time_point<std::chrono::steady_clock> t1 = std::chrono::steady_clock::now();
    //         std::chrono::duration<double, std::milli> elapsed = t1 - t0;
    //         printf("[done] %lf (ms)\n", elapsed.count());
    // #if defined(WITNESS_GEN)
    //         cudaDeviceSynchronize();
    //         // printV<<<1, 1>>>();
    //         printVH();
    //         cudaDeviceSynchronize();
    // #endif
    //         exit(0);
    //     }

    // __global__ void bitFlip(int i)
    // {
    //     valQ_C[i] = !valQ_C[i];
    // }

    // void q_round()
    // {
    //     init_ra<<<Nb, Nt, 0, sA>>>();
    //     if (dts0 == NULL)
    //     {
    //         cub::DeviceRadixSort::SortPairsDescending(dts0, tsb, d_h, d_h_s, d_hi, d_hi_s, Nk, 0, 32, sA);
    //         ccE(cudaMallocAsync(&dts0, tsb, sA));
    //     }
    //     cub::DeviceRadixSort::SortPairsDescending(dts0, tsb, d_h, d_h_s, d_hi, d_hi_s, Nk, 0, 32, sA);
    //     int hidx;
    //     ccE(cudaMemcpyAsync(&hidx, d_hi_s, 4, cudaMemcpyDeviceToHost, sA));
    //     ccE(cudaStreamSynchronize(sA));
    //     ccE(cudaMemcpyAsync(valQ, valS + h_vn * hidx, h_vn * sizeof(bool), cudaMemcpyDeviceToDevice, sA));
    //     ccE(cudaMemcpyAsync(&hq, d_h_s + hidx, sizeof(int), cudaMemcpyDeviceToHost, sA));
    //     // BEGIN DEBUG
    //     cudaDeviceSynchronize();
    //     if (hq == h_cn)
    //     {
    //         hmax = hq;
    //         Q2M();
    //         terminate();
    //     }
    //     if(hq > hmax){
    //         hmax = hq;
    //         Q2M(sA);
    //     }
    // }

    // void localsearch()
    // {
    //     std::chrono::time_point<std::chrono::steady_clock> t1 = std::chrono::steady_clock::now();
    //     std::chrono::duration<double, std::milli> elapsed = t1 - t0;
    //     ccE(cudaStreamSynchronize(sA));
    //     int jz = 0;
    //     while (hmax < h_cn)
    //     {
    //         jz++;
    //         ccE(cudaMemsetAsync(bVS, 0, h_cn * h_vn * sizeof(bool), sB)); // COMMENTX
    //         lKern<<<bszL, tszL, 0, sB>>>();
    //         if (dtsS == NULL)
    //         {
    //             cub::DeviceSegmentedReduce::Sum(dtsS, tsb, bVS, d_hN, h_vn, d_cOff, d_cOff + 1, sB);
    //             // printf("REDUCESUM [%d] : %d\n",h_vn,tsb);
    //             ccE(cudaMallocAsync(&dtsS, tsb, sB));
    //         }
    //         cub::DeviceSegmentedReduce::Sum(dtsS, tsb, bVS, d_hN, h_vn, d_cOff, d_cOff + 1, sB);
    //         ccE(cudaStreamSynchronize(sB));
    //         // ccE(cudaMemsetAsync(bVS, 0, h_cn * h_vn * sizeof(bool), sA));
    //         if (dtsA == NULL)
    //         {
    //             cub::DeviceReduce::ArgMax(dtsA, tsb, d_hN, d_argmin, h_vn, sB);
    //             // printf("ARGMAX [%d] : %d\n",h_vn,tsb);
    //             ccE(cudaMallocAsync(&dtsA, tsb, sB));
    //         }
    //         cub::DeviceReduce::ArgMax(dtsA, tsb, d_hN, d_argmin, h_vn, sB);
    //         int new_h;
    //         int new_hidx;
    //         ccE(cudaMemcpyAsync(&new_h, &d_argmin[0].value, sizeof(int), cudaMemcpyDeviceToHost, sB));
    //         ccE(cudaMemcpyAsync(&new_hidx, &d_argmin[0].key, sizeof(int), cudaMemcpyDeviceToHost, sA));
    //         cudaStreamSynchronize(sB);
    //         // cudaStreamSynchronize(sA); // COMMENTX
    //         // pN<<<1,1>>>(d_hN,h_vn);
    //         // update state
    //         if (new_h > hq)
    //         {
    //             bitFlip<<<1, 1, 0, sA>>>(new_hidx);
    //             cudaStreamSynchronize(sA);
    //             hq = new_h;
    //             if (hq > hmax)
    //             {
    //                 hmax = hq;
    //                 Q2M(sB);
    //             }
    //             // printf("FLIP %d\n",new_hidx);
    //             if (hq == h_cn)
    //             {
    //                 // printf("POST_CHECK:\n");
    //                 // postChk<<<1,1>>>();
    //                 cudaDeviceSynchronize();
    //                 terminate();
    //             }
    //         }
    //         else
    //         {
    //             printf(".\n");
    //             cur_idx++;
    //             if (cur_idx >= Nk)
    //             {
    //                 cur_idx = 0;
    //                 q_round();
    //             }else{
    //             int hidx;
    //             ccE(cudaMemcpyAsync(&hidx, d_hi_s + cur_idx, 4, cudaMemcpyDeviceToHost, sB));
    //             ccE(cudaMemcpyAsync(&hq, d_h_s + cur_idx, 4, cudaMemcpyDeviceToHost, sB));
    //             ccE(cudaMemcpyAsync(valQ, valS + h_vn * hidx, h_vn * sizeof(bool), cudaMemcpyDeviceToDevice, sB));
    //             }
    //         }
    //         t1 = std::chrono::steady_clock::now();
    //         elapsed = t1 - t0;
    //         cudaDeviceSynchronize();
    //         printf("%.3lf (ms) | new_h: %d\n", elapsed.count(), new_h);
    //         if (elapsed.count() > TIMEOUT)
    //         {
    //             printf("[TIMEOUT] at %.6lf%% [%d / %d]\n",
    //                    100.0 * ((double)hmax / (double)h_cn), hmax, h_cn);
    //             break;
    //         }
    //     }
    //     terminate();
    // }

    void initST()
    {
// cudaDeviceScheduleBlockingSync or:
/*
:in ascending order from Low latency to High latency:
CudaDeviceScheduleSpin,
CudaDeviceScheduleYield,
cudaDeviceScheduleBlockingSync
*/
// ccE(cudaSetDeviceFlags(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)));
#pragma unroll(QS)
        for (auto i = 0; i < QS; i++)
        {
            ccE(cudaStreamCreateWithFlags(&(Qs[i].s), cudaStreamNonBlocking));
            ccE(cudaMallocAsync(&(Qs[i].d_argmin), sizeof(cub::KeyValuePair<int, int>), Qs[i].s));
        }
    }

    inline uint64_t get_T64()
    {
        return std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
    }

    __global__ void initVQ(uint8_t __restrict__ *v, curandState_t __restrict__ *cst)
    {
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tidx >= c_vn)
            return;
        v[tidx] = curand(cst + tidx);
    }

    __global__ void initCQ(curandState_t __restrict__ *cst, uint64_t seed, uint64_t off = 0)
    {
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        curand_init(tidx + seed + off, 0, 0, cst + tidx);
    }

    void initCM0()
    {
        vsize = (7 + h_vn) / 8;
        kl_init(bszI, tszI, vsize);
#pragma unroll (QS)
        for (auto i = 0; i < QS; i++)
        {
            ccE(cudaMallocAsync(&(Qs[i].cst), sizeof(curandState_t) * vsize, Qs[i].s));
            initCQ<<<bszI, tszI, 0, Qs[i].s>>>(Qs[i].cst, get_T64(), 0xffffff*i);
            ccE(cudaMallocAsync(&(Qs[i].hN), sizeof(uint32_t) * 2 * (h_vn + 1), Qs[i].s));
            ccE(cudaMallocAsync(&(Qs[i].valQ), sizeof(uint8_t) * vsize, Qs[i].s));
            ccE(cudaMallocAsync(&(Qs[i].valM), sizeof(uint8_t) * vsize, Qs[i].s));
            ccE(cudaMallocAsync(&(Qs[i].flg), sizeof(uint32_t) * h_cn, Qs[i].s));
        }

        // BATCH SIZES?
        // checking for memory?
        // size_t free, total;
        // ccE(cudaMemGetInfo(&free,&total));
        // ccE(cudaMallocAsync(&bVS, sizeof(bool) * h_cn * h_vn, sA));
        ccE(cudaMemcpyToSymbolAsync(c_vn, &h_vn, sizeof(int), 0, cudaMemcpyHostToDevice, Qs[0].s));
        ccE(cudaMemcpyToSymbolAsync(c_cn, &h_cn, sizeof(int), 0, cudaMemcpyHostToDevice, Qs[0].s));
        #pragma unroll (QS)
        for (auto i = 0; i < QS; i++)
        {
            initVQ<<<bszI,tszI,0,Qs[i].s>>>(Qs[i].valQ,Qs[i].cst);
        }
        ccE(cudaMallocAsync(&d_indxs, sizeof(int) * (h_cn + 1), Qs[0].s));
        // ccE(cudaMemcpyToSymbolAsync(d_indxs_C, &d_indxs, sizeof(int *), 0, cudaMemcpyHostToDevice, sA));
        // ccE(cudaMallocAsync(&d_cOff, sizeof(int) * (h_vn + 1), sA));
        // initO<<<bszV, tszV, 0, sA>>>(d_cOff);
    }

    void initCM()
    {
        ccE(cudaMemcpyToSymbolAsync(c_ln, &h_ln, sizeof(int), 0, cudaMemcpyHostToDevice, Qs[0].s));
        kl_LKX(bszL, tszL);
        // LATER: statically tune and optimize these parameters
        // bszL = (h_ln + 511) / 512;
        // // tszL = (h_ln + bszL - 1) / bszL;
        // tszL = 512;
        bszC = (h_cn + 511) / 512;
        // // tszC = (h_cn + bszC - 1) / bszC;
        tszC = 512;
        ccE(cudaMallocAsync(&d_lits, sizeof(int) * h_ln, Qs[0].s));
        ccE(cudaMemcpyAsync(d_lits, lits, sizeof(int) * h_ln, cudaMemcpyHostToDevice, Qs[0].s));
        ccE(cudaMallocAsync(&d_L2C, sizeof(int) * h_ln, Qs[0].s));
        ccE(cudaMemcpyToSymbolAsync(c_lits, &d_lits, sizeof(int *), 0, cudaMemcpyHostToDevice, Qs[0].s));
        ccE(cudaMemcpyToSymbolAsync(c_L2C, &d_L2C, sizeof(int *), 0, cudaMemcpyHostToDevice, Qs[0].s));
        ccE(cudaMemcpyAsync(d_indxs, indxs, sizeof(int) * (h_cn + 1), cudaMemcpyHostToDevice, Qs[1].s));
    }

    __global__ void initL2C(const int *__restrict__ indxs)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= c_cn)
            return;
        int i0 = indxs[idx];
        int i1 = indxs[idx + 1];
        for (int i = i0; i < i1; i++)
        {
            c_L2C[i] = idx;
        }
    }

    void printV(const uint8_t *__restrict__ vals, bool flipped = false)
    {
        printf("V:");
        for (int i = 0; i < vsize; i++)
        {
            printf("%d", (vals[i] & 1) ^ flipped);
            printf("%d", ((vals[i] & 2) >> 1) ^ flipped);
            printf("%d", ((vals[i] & 4) >> 2) ^ flipped);
            printf("%d", ((vals[i] & 8) >> 3) ^ flipped);
            printf("%d", ((vals[i] & 16) >> 4) ^ flipped);
            printf("%d", ((vals[i] & 32) >> 5) ^ flipped);
            printf("%d", ((vals[i] & 64) >> 6) ^ flipped);
            printf("%d", ((vals[i] & 128) >> 7) ^ flipped);
        }
        printf("\n");
    }

    template <int S>
    __global__ void LKX(const uint8_t *__restrict__ vals, bool flipped, const uint32_t *__restrict__ flg, uint32_t *__restrict__ h)
    {
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x;
        // can this be assert?
        if (tidx * S >= c_ln)
        {
            return;
        }
        // future idea: lookahead at L2C of last literal
        for (int i = 0; i < S; i++)
        {
            int lidx = tidx + i * stride;
            if (lidx >= c_ln)
                break;
            int cli = c_L2C[lidx];
            if (flg[cli] == (ST | UT))
            {
                continue;
            }
            int vj = c_lits[lidx];
            int vja = abs(vj) - 1;
            bool v = (vals[vja / 8] & (1 << (vja % 8)) > 0) ^ flipped;
            // if(flipped) v = !v;
            int vn = (vj > 0 ? 1 : 0);
            int r = !(vn ^ v);
            if (r)
            {
                uint32_t flgO = atomicOr((unsigned int *)(flg + cli), ST);
                if ((flgO & ST) != 0)
                {
                    continue;
                }
                atomicAdd((unsigned int *)h, 1);
                atomicAdd((unsigned int *)(h + c_vn + 2 + vja), 1);
                for (int vidx = 0; vidx < c_vn; vidx++)
                {
                    if (vidx == vja)
                    {
                        continue;
                    }
                    atomicAdd((unsigned int *)(h + 1 + vidx), 1);
                }
            }
            else
            {
                uint32_t flgO = atomicOr((unsigned int *)(flg + cli), UT);
                if ((flgO & UT) != 0)
                {
                    continue;
                }
                atomicAdd((unsigned int *)(h + c_vn + 1), 1);
                atomicAdd((unsigned int *)(h + 1 + vja), 1);
                for (int vidx = 0; vidx < c_vn; vidx++)
                {
                    if (vidx == vja)
                    {
                        continue;
                    }
                    atomicAdd((unsigned int *)(h + c_vn + 2 + vidx), 1);
                }
            }
        }
    }

    __global__ void bit_flip(uint8_t *__restrict__ v, int i)
    {
        v[i] = !v[i];
    }

    inline void updateQ(Q_data &Q, int idxM)
    {
        if (idxM == 0)
        {
            return;
        }
        if (idxM == h_vn + 1)
        {
            Q.vQF = !Q.vQF;
            return;
        }
        if(idxM > (h_vn+1)){
            Q.vQF = !Q.vQF;
        }
        bit_flip<<<1, 1, 0, Q.s>>>(Q.valQ, (idxM % (h_vn + 1)) - 1);
    }

    inline void terminate_sat(uint8_t *v, bool flipped, cudaStream_t s)
    {
        g_lock.lock();
        std::chrono::time_point<std::chrono::steady_clock> t1 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> el = t1 - t0;
        dprint("[SAT] : %.3lf (ms)\n", el.count());
        uint8_t *h_v = (uint8_t *)malloc(sizeof(uint8_t) * vsize);
        ccE(cudaMemcpyAsync(h_v, v, vsize, cudaMemcpyDeviceToHost, s));
        ccE(cudaStreamSynchronize(s));
        printV(h_v, flipped);
        exit(0);
        // g_lock.unlock(); // not really needed
    }

    inline void terminate_sat(Q_data &Q)
    {
        if (Q.doCpy)
        {
            terminate_sat(Q.valQ, Q.vQF, Q.s);
        }
        else
        {
            terminate_sat(Q.valM, Q.vMF, Q.s);
        }
    }

    inline void terminate(Q_data &Q)
    {
        g_lock.lock();
        std::chrono::time_point<std::chrono::steady_clock> t1 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> el = t1 - t0;
        printf("[%d / %d] @ %.3lf (ms)\n", Q.hMax, h_cn, el.count());
        bool flipped = Q.vMF;
        uint8_t *v = Q.valM;
        if (Q.doCpy)
        {
            flipped = Q.vQF;
            v = Q.valQ;
        }
        uint8_t *h_v = (uint8_t *)malloc(sizeof(uint8_t) * vsize);
        ccE(cudaMemcpyAsync(h_v, v, vsize, cudaMemcpyDeviceToHost, Q.s));
        ccE(cudaStreamSynchronize(Q.s));
        printV(h_v, flipped);
        exit(0);
        // g_lock.unlock(); // redundant as we kill process
    }

    void ls(Q_data &Q)
    {
        std::chrono::time_point<std::chrono::steady_clock> t1 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t1 - t0;
        size_t tsb = 0;
        void *temp_s = NULL;
        bool timeout = false;
        while (Q.hMax < h_cn)
        {
            ccE(cudaMemsetAsync(Q.flg, 0, sizeof(uint32_t) * h_cn, Q.s));
            ccE(cudaMemsetAsync(Q.hN, 0, sizeof(uint32_t) * 2 * (h_vn + 1), Q.s));
            // std::chrono::time_point<std::chrono::steady_clock> tx0 = std::chrono::steady_clock::now();
            LKX<SX><<<bszL, tszL, 0, Q.s>>>(Q.valQ, Q.vQF, Q.flg, Q.hN);
            if (temp_s == NULL)
            {
                cub::DeviceReduce::ArgMax(temp_s, tsb, Q.hN, Q.d_argmin, 2 * (h_vn + 1), Q.s);
                ccE(cudaMallocAsync(&temp_s, tsb, Q.s));
            }
            cub::DeviceReduce::ArgMax(temp_s, tsb, Q.hN, Q.d_argmin, 2 * (h_vn + 1), Q.s);
            int new_h;
            int new_hidx;
            ccE(cudaMemcpyAsync(&new_h, &Q.d_argmin[0].value, sizeof(int), cudaMemcpyDeviceToHost, Q.s));
            ccE(cudaMemcpyAsync(&new_hidx, &Q.d_argmin[0].key, sizeof(int), cudaMemcpyDeviceToHost, Q.s));
            ccE(cudaStreamSynchronize(Q.s));
            bool going_up = new_h > Q.hQ;
            // printf("%d @ %d\n",new_h, new_hidx);
            if (going_up)
            {
                // printf(".");
                Q.hQ = new_h;
                updateQ(Q, new_hidx);
                // update vQ
                if (new_h > Q.hMax)
                {
                    Q.hMax = new_h;
                    Q.doCpy = true;
                }
                // is hidx == 0?
            }
            if (new_hidx == 0 || (!going_up))
            {
                // doCpy -> create new Q
                if (Q.doCpy)
                {
                    Q.vMF = Q.vQF;
                    Q.doCpy = false;
                    cudaMemcpyAsync(Q.valM, Q.valQ, vsize, cudaMemcpyDeviceToDevice, Q.s);
                }
                Q.vQF = false;
                Q.hQ = 0;
                // printf("O");
                initVQ<<<bszI, tszI, 0, Q.s>>>(Q.valQ, Q.cst);
            }
            // std::chrono::time_point<std::chrono::steady_clock> tx1 = std::chrono::steady_clock::now();
            // std::chrono::duration<double, std::micro> el = tx1 - tx0;
            // printf("%lf (us)\n", el.count());
            t1 = std::chrono::steady_clock::now();
            elapsed = t1 - t0;
            if (elapsed.count() > TIMEOUT)
            {
                timeout = true;
                break;
            }
        }
        if (!timeout)
            terminate(Q);
        // return; // redundant as terminate exit(0)s
    }

    uint32_t solve(bool ires, bool *&res)
    {
        dprint("solving...\n");
        t0 = std::chrono::steady_clock::now();
        ccE(cudaDeviceSynchronize());
        initL2C<<<bszC, tszC, 0, Qs[0].s>>>(d_indxs); // T ~ 20 (us)
        cudaFreeAsync(d_indxs, Qs[1].s);
        cudaStreamSynchronize(Qs[0].s);
        std::thread ts[QS];

        #pragma unroll (QS)
        for (auto i = 0; i < QS; i++)
        {
            ts[i] = std::thread(ls,std::ref(Qs[i]));
        }
        
          #pragma unroll (QS)
        for (auto i = 0; i < QS; i++)
        {
            ts[i].join();
        }
        int maxI = -1;
        int maxH = -1;
        for(auto i = 0; i<QS; i++){
            int hm = Qs[i].hMax;
            if(hm > maxH){
                maxH = hm;
                maxI = i;
            }
        }
        terminate(Qs[maxI]);
        // q_round();
        // localsearch();
    }
}