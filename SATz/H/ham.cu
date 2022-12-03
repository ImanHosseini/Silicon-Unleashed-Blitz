#include "melkor.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <chrono>
#include <cuda_profiler_api.h>
#include <thread>
#include <math_constants.h>

#define HAM_BETA 1
#define HAM_2BTA 2 * HAM_BETA + 1

#ifdef HAM_DOUBLE
typedef double angle_t;
#define HAM_DT 0.001
#define HAM_PI CUDART_PI_F
#define HAM_M 1.0
#define HAM_NZ 100.0
#else
typedef float angle_t;
#define HAM_DT 0.001f
#define HAM_PI CUDART_PI
#define HAM_M 1.0f
#define HAM_NZ 50000.0f
#endif

#define HAM_DMP 0.9998

// starting mean momentum
#define PJ HAM_PI *HAM_M / (HAM_DT * HAM_NZ)
// dt over m
#define HAM_DTOM HAM_DT / HAM_M

#define TIMEOUT 50000 * 100

namespace Melkor
{
    std::mutex g_lock;

    typedef struct Q_data
    {
        int Qnumber;
        cudaStream_t s;
        angle_t *_q;
        angle_t *_p;
        angle_t *_r;
        angle_t *_dHdq;
        // int32_t *_cmx;
        angle_t *_lV;
        bool *dvs;
        uint8_t *satf;
        uint32_t cost = 1000;
        uint32_t *d_sat_ptr = NULL;
        uint32_t h_sat = 0;
        uint64_t loops = 0;
        curandState_t *cst;
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
    int *indxs;
    int *lits;
    int *d_lits;
    int *d_indxs;

    __constant__ int c_ln;
    __constant__ uint32_t c_vn;
    __constant__ uint32_t c_cn;

    /*---- Time Keeping ------------------------------------*/
    std::chrono::time_point<std::chrono::steady_clock> t0;

    dim3 bszL;
    dim3 tszL;
    dim3 bszS;
    dim3 tszS;
    dim3 bszC;
    dim3 tszC;
    // For initialization, related to CQF
    uint32_t bszI;
    uint32_t tszI;
    // For h_vn
    uint32_t bszV;
    uint32_t tszV;
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
        // printf("%s:[", tag);
        for (int i = 0; i < n; i++)
        {
            if (std::is_integral<T>::value)
            {
                printf("%d ", p[i]);
            }
            else if (std::is_floating_point<T>::value)
            {
                printf("%f ", p[i]);
            }
        }
        printf("]\n");
    }

    // for debugging purposes!
    __global__ void chk2(uint8_t *vals, int hv, bool flipped, bool *setf)
    {
        // bool* setf = (bool*) malloc(sizeof(bool)*c_cn);
        int s = 0;
        for (int i = 0; i < c_ln; i++)
        {
            int cli = c_L2C[i];
            int vj = c_lits[i];
            int vja = abs(vj) - 1;
            int vn = (vj > 0 ? 1 : 0);
            bool v = (vals[(vja / 8)] & (1 << (vja % 8))) > 0;
            v = v ^ flipped;
            int r = !(vn ^ v);
            if (r)
            {
                if (setf[cli])
                    continue;
                setf[cli] = true;
                s += 1;
            }
        }
        if (hv != s)
        {
            printf("sat: %d - h: %d\n", s, hv);
        }
    }

    // for debugging purposes!
    __global__ void chk(uint8_t *vals, int hv, bool flipped = false)
    {
        bool *setf = (bool *)malloc(sizeof(bool) * c_cn);
        int s = 0;
        for (int i = 0; i < c_ln; i++)
        {
            int cli = c_L2C[i];
            int vj = c_lits[i];
            int vja = abs(vj) - 1;
            int vn = (vj > 0 ? 1 : 0);
            bool v = (vals[(vja / 8)] & (1 << (vja % 8))) > 0;
            v = v ^ flipped;
            int r = !(vn ^ v);
            if (r)
            {
                if (setf[cli])
                    continue;
                setf[cli] = true;
                s += 1;
            }
        }
        assert(hv == s);
        free(setf);
        // printf("H:%d\n",s);
    }

    template <int S>
    __global__ void initVQS(angle_t __restrict__ *v, curandState_t __restrict__ *cst, angle_t mean = 0.0f)
    {
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = 0; i < S; i++)
        {
            int j = tidx + i * stride;
            if (j >= c_vn)
                break;
#ifdef HAM_DOUBLE
            v[j] = remainder((curand_normal_double(cst + tidx) + mean) * HAM_PI, (angle_t)(2.0 * HAM_PI));
#else
            v[j] = remainder((curand_normal(cst + tidx) + mean) * HAM_PI, (angle_t)(2.0f * HAM_PI));
#endif
        }
    }

    template <int S>
    __global__ void initVQP(angle_t __restrict__ *v, curandState_t __restrict__ *cst, angle_t stddv = PJ)
    {
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = 0; i < S; i++)
        {
            int j = tidx + i * stride;
            if (j >= c_vn)
                break;
#ifdef HAM_DOUBLE
            v[j] = (curand_normal_double(cst + tidx) + mean) * mean * 0.5;
#else
            v[j] = curand_normal(cst + tidx) * stddv;
#endif
        }
    }

    __global__ void update_lV(const angle_t __restrict__ *r, angle_t __restrict__ *lV)
    {
        uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
        uint32_t tidx = bid * blockDim.x + threadIdx.x;
        if (tidx >= c_ln)
        {
            return;
        }
        // int cli = c_L2C[tidx];
        int l = c_lits[tidx];
        int ri = abs(l) - 1;
        int rval = r[ri];
        // assert(!isnan(rval));
        if (l < 0)
            rval = -r[ri];
        lV[tidx] = rval;
    }

    __global__ void qtor(const angle_t __restrict__ *q, angle_t __restrict__ *r)
    {
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tidx >= c_vn)
            return;
        angle_t cop = cos(q[tidx] - HAM_PI * 0.5);
        angle_t v = pow(cop, (int)HAM_2BTA);
        r[tidx] = v;
    }

    __global__ void update_qr(angle_t __restrict__ *q, angle_t __restrict__ *r, const angle_t __restrict__ *p)
    {
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tidx >= c_vn)
            return;
        // Euler step
        angle_t qn = fma((angle_t)HAM_DTOM, p[tidx], q[tidx]);
        if (isnan(qn))
        {
            printf("q[tidx]:%f p[tidx]:%f\n", q[tidx], p[tidx]);
        }
        assert(!isnan(qn));
        qn = remainder(qn, (angle_t)(2.0 * HAM_PI));
        q[tidx] = qn;
        if (isnan(qn))
        {
            printf("###: q[tidx]:%f p[tidx]:%f\n", q[tidx], p[tidx]);
        }
        assert(!isnan(qn));
        r[tidx] = pow(cos(qn - HAM_PI * 0.5), (int)HAM_2BTA);
    }

    __global__ void update_p(angle_t __restrict__ *p, const angle_t __restrict__ *dHdq, angle_t df)
    {
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tidx >= c_vn)
            return;
        // Euler step
        // assert( !isnan(dHdq[tidx]));
        angle_t pn = fma((angle_t)HAM_DT, dHdq[tidx], p[tidx]);
        if (isinf(pn))
        {
            printf("%f %f\n", dHdq[tidx], p[tidx]);
        }
        assert(!isinf(pn));
        p[tidx] = pn * df;
    }

    inline int kl_init(uint32_t &bsz, uint32_t &tsz, int wpt = 1)
    {
        int k = (h_vn + wpt - 1) / wpt;
        bsz = (k + 511) / 512;
        tsz = (k + bsz - 1) / bsz;
        dprint("kernel<%d><<<%d,%d>>>\n", wpt, bsz, tsz);
        return k;
    }

    inline void kl_LKX(dim3 &bsz, dim3 &tsz, uint32_t items)
    {
        // printf("items: %d\n",items);
        uint32_t bzt = (items + 511) / 512;
        // printf("= %d x 512\n",bzt);
        int q = (bzt + 65534) / 65535;
        // printf("bzt = %d x 65535\n",q);
        int r = (bzt + q - 1) / q;
        bsz = dim3(r, q);
        int j = q * r;
        int tz = (items + j - 1) / j;
        tsz = dim3(tz);
        dprint("update_LKX<<<(%d,%d),%d>>>\n", bsz.x, bsz.y, tsz.x);
    }

    // inline void kl_LKS(dim3 &bsz, dim3 &tsz)
    // {
    //     uint32_t a = h_ln * (SE / WS);
    //     uint32_t bzt = (a + 511) / 512;
    //     tsz = dim3(512);
    //     if (bzt <= 65535)
    //     {
    //         bsz = dim3(bzt);
    //     }
    //     else
    //     {
    //         int q = (bzt + 65534) / 65535;
    //         int r = (bzt + q - 1) / q;
    //         bsz = dim3(q, r);
    //     }
    //     dprint("LKS<<<(%d,%d),%d>>>\n", bsz.x, bsz.y, tsz);
    // }

    // void printV(const uint8_t *__restrict__ vals, bool flipped = false)
    // {
    //     printf("V:");
    //     for (int i = 0; i < vsize; i++)
    //     {
    //         printf("%d", ((vals[i] >> 0) & 1) ^ flipped);
    //         printf("%d", ((vals[i] >> 1) & 1) ^ flipped);
    //         printf("%d", ((vals[i] >> 2) & 1) ^ flipped);
    //         printf("%d", ((vals[i] >> 3) & 1) ^ flipped);
    //         printf("%d", ((vals[i] >> 4) & 1) ^ flipped);
    //         printf("%d", ((vals[i] >> 5) & 1) ^ flipped);
    //         printf("%d", ((vals[i] >> 6) & 1) ^ flipped);
    //         printf("%d", ((vals[i] >> 7) & 1) ^ flipped);
    //     }
    //     printf("\n");
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
            Qs[i].Qnumber = i;
            ccE(cudaStreamCreateWithFlags(&(Qs[i].s), cudaStreamNonBlocking));
            ccE(cudaMallocAsync(&Qs[i].d_sat_ptr, sizeof(uint32_t), Qs[i].s));
        }
    }

    inline uint64_t get_T64()
    {
        return std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
    }

    __global__ void initCQ(curandState_t __restrict__ *cst, uint64_t seed, uint64_t off, int k)
    {
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tidx >= k)
            return;
        curand_init(tidx + seed + off, 0, 0, cst + tidx);
    }

    void initCM0()
    {
        ccE(cudaMemcpyToSymbolAsync(c_vn, &h_vn, sizeof(int), 0, cudaMemcpyHostToDevice, Qs[0].s));
        ccE(cudaMemcpyToSymbolAsync(c_cn, &h_cn, sizeof(int), 0, cudaMemcpyHostToDevice, Qs[0].s));
        int k = kl_init(bszI, tszI, CQF);
        kl_init(bszV, tszV);
        kl_LKX(bszC, tszC, h_cn);
#pragma unroll(QS)
        for (auto i = 0; i < QS; i++)
        {
            ccE(cudaMallocAsync(&(Qs[i].cst), sizeof(curandState_t) * k, Qs[i].s));
            // initCQ<<<bszI, tszI, 0, Qs[i].s>>>(Qs[i].cst, get_T64(), 0xffff * i, k);
            initCQ<<<bszI, tszI, 0, Qs[i].s>>>(Qs[i].cst, 0, 0xffff * i, k);
            ccE(cudaMallocAsync(&(Qs[i]._q), sizeof(angle_t) * h_vn, Qs[i].s));
            initVQS<CQF><<<bszI, tszI, 0, Qs[i].s>>>(Qs[i]._q, Qs[i].cst);
            ccE(cudaMallocAsync(&(Qs[i]._p), sizeof(angle_t) * h_vn, Qs[i].s));
            initVQP<CQF><<<bszI, tszI, 0, Qs[i].s>>>(Qs[i]._p, Qs[i].cst);
            // cudaDeviceSynchronize();
            // pN<<<1,1>>>(Qs[i]._p, 1672);
            //  cudaDeviceSynchronize();
            // exit(0);
            ccE(cudaMallocAsync(&(Qs[i].dvs), sizeof(bool) * h_vn, Qs[i].s));
            ccE(cudaMallocAsync(&(Qs[i].satf), sizeof(uint8_t) * h_cn, Qs[i].s));
            ccE(cudaMallocAsync(&(Qs[i]._r), sizeof(angle_t) * h_vn, Qs[i].s));
            ccE(cudaMallocAsync(&(Qs[i]._r), sizeof(angle_t) * h_vn, Qs[i].s));
            ccE(cudaMallocAsync(&(Qs[i]._dHdq), sizeof(angle_t) * h_vn, Qs[i].s));
        }

#pragma unroll(QS)
        for (auto i = 0; i < QS; i++)
        {
            qtor<<<bszV, tszV, 0, Qs[i].s>>>(Qs[i]._q, Qs[i]._r);
        }
        ccE(cudaMallocAsync(&d_indxs, sizeof(int) * (h_cn + 1), Qs[0].s));
    }

    void initCM()
    {
        ccE(cudaMemcpyToSymbolAsync(c_ln, &h_ln, sizeof(int), 0, cudaMemcpyHostToDevice, Qs[0].s));
        kl_LKX(bszL, tszL, h_ln);
        // kl_LKS(bszS, tszS);
        // bszC = (h_cn + 511) / 512;
        // tszC = 512;
        ccE(cudaMallocAsync(&d_lits, sizeof(int) * h_ln, Qs[0].s));
        ccE(cudaMemcpyAsync(d_lits, lits, sizeof(int) * h_ln, cudaMemcpyHostToDevice, Qs[0].s));
        ccE(cudaMallocAsync(&d_L2C, sizeof(int) * h_ln, Qs[0].s));
        ccE(cudaMemcpyToSymbolAsync(c_lits, &d_lits, sizeof(int *), 0, cudaMemcpyHostToDevice, Qs[0].s));
        ccE(cudaMemcpyToSymbolAsync(c_L2C, &d_L2C, sizeof(int *), 0, cudaMemcpyHostToDevice, Qs[0].s));
        ccE(cudaMemcpyAsync(d_indxs, indxs, sizeof(int) * (h_cn + 1), cudaMemcpyHostToDevice, Qs[1].s));
#pragma unroll(QS)
        for (auto i = 0; i < QS; i++)
        {
            ccE(cudaMallocAsync(&Qs[i]._lV, sizeof(angle_t) * h_ln, Qs[i].s));
        }
    }

    __global__ void initL2C(const int *__restrict__ indxs)
    {
        int bid = blockIdx.y * gridDim.x + blockIdx.x;
        int idx = bid * blockDim.x + threadIdx.x;
        // int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= c_cn)
            return;
        int i0 = indxs[idx];
        int i1 = indxs[idx + 1];
        for (int i = i0; i < i1; i++)
        {
            c_L2C[i] = idx;
        }
    }

    // inline void terminate(Q_data &Q)
    // {
    //     g_lock.lock();
    //     std::chrono::time_point<std::chrono::steady_clock> t1 = std::chrono::steady_clock::now();
    //     std::chrono::duration<double, std::milli> el = t1 - t0;
    //     printf("[%d / %d] @ %.3lf (ms)\n", Q.hMax, h_cn, el.count());
    //     bool flipped = Q.vMF;
    //     uint8_t *v = Q.valM;
    //     if (Q.doCpy)
    //     {
    //         flipped = Q.vQF;
    //         v = Q.valQ;
    //     }
    //     uint8_t *h_v = (uint8_t *)malloc(sizeof(uint8_t) * vsize);
    //     ccE(cudaMemcpyAsync(h_v, v, vsize, cudaMemcpyDeviceToHost, Q.s));
    //     ccE(cudaStreamSynchronize(Q.s));
    //     printV(h_v, flipped);
    //     exit(0);
    //     // g_lock.unlock(); // redundant as we kill process
    // }

    __global__ void discretize(const __restrict__ angle_t *q, bool *vals)
    {
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tidx >= c_vn)
            return;
        angle_t a = q[tidx];
        bool spin = false;
        if (a > (angle_t)0.0)
        {
            spin = true;
        }
        vals[tidx] = spin;
    }

    __global__ void calc_sat_count(const bool *vals, __restrict__ uint32_t *csat)
    {
        uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
        uint32_t tidx = bid * blockDim.x + threadIdx.x;
        if (tidx >= c_ln)
        {
            return;
        }
        int l = c_lits[tidx];
        int ri = abs(l) - 1;
        int rv = vals[ri];
        bool sat = !(rv ^ (l > 0));
        if (sat)
        {
            int cli = c_L2C[tidx];
            atomicInc(csat + cli, c_cn);
        }
    }

    __global__ void calc_satf(const bool *vals, __restrict__ uint8_t *satf)
    {
        uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
        uint32_t tidx = bid * blockDim.x + threadIdx.x;
        if (tidx >= c_ln)
        {
            return;
        }
        int l = c_lits[tidx];
        int ri = abs(l) - 1;
        bool rv = vals[ri];
        bool sat = !(rv ^ (l > 0));
        if (sat)
        {
            int cli = c_L2C[tidx];
            satf[cli] = 1;
        }
    }

    void printV(Q_data &Q)
    {
        bool *v = (bool *)malloc(sizeof(bool) * h_vn);
        ccE(cudaMemcpyAsync(v, Q.dvs, sizeof(bool) * h_vn, cudaMemcpyDeviceToHost, Q.s));
        ccE(cudaStreamSynchronize(Q.s));
        g_lock.lock();
        printf("Q%d:\n", Q.Qnumber);
        for (auto i = 0; i < h_vn; i++)
        {
            printf("%d", v[i]);
        }
        printf("\n");
        free(v);
        g_lock.unlock();
    }

    uint32_t calc_cost(Q_data &Q, void *&tmps, size_t &tsb)
    {
        discretize<<<bszV, tszV, 0, Q.s>>>(Q._q, Q.dvs);
        ccE(cudaMemsetAsync(Q.satf, 0, sizeof(uint8_t) * h_cn, Q.s));
        calc_satf<<<bszL, tszL, 0, Q.s>>>(Q.dvs, Q.satf);
        if (tmps == NULL)
        {
            cub::DeviceReduce::Sum(tmps, tsb, Q.satf, Q.d_sat_ptr, h_cn, Q.s);
            ccE(cudaMallocAsync(&tmps, tsb, Q.s));
        }
        cub::DeviceReduce::Sum(tmps, tsb, Q.satf, Q.d_sat_ptr, h_cn, Q.s);
        // pN<<<1,1>>>(Q.satf + 5000, 100);
        // cudaDeviceSynchronize();
        // printV(Q);
        ccE(cudaMemcpyAsync(&Q.cost, Q.d_sat_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, Q.s));
        ccE(cudaStreamSynchronize(Q.s));
        // printf("sats: %d\n",Q.cost);
        uint32_t cost = h_cn - Q.cost;
        Q.cost = cost;
        return cost;
    }

    __global__ void calc_dHdq(const cub::KeyValuePair<int, angle_t> *maxlit,
                              const __restrict__ angle_t *q,
                              const __restrict__ angle_t *r,
                              __restrict__ angle_t *dHdq,
                              const int *indxs)
    {
        uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
        uint32_t tidx = bid * blockDim.x + threadIdx.x;
        if (tidx >= c_cn)
        {
            return;
        }
        cub::KeyValuePair<int, angle_t> kv = maxlit[tidx];
        int off = kv.key;
        // angle_t rval = kv.value;
        int lidx = indxs[tidx] + off;
        // test
        int i0 = indxs[tidx];
        int i1 = indxs[tidx + 1];
        for (auto i = i0; i < i1; i++)
        {
            int l = c_lits[lidx];
            int qi = abs(l) - 1;
            angle_t spinA = HAM_PI * 0.5;
            if (l < 0)
            {
                spinA = HAM_PI * 1.5;
            }
            angle_t derivative = ((angle_t)HAM_2BTA) * pow(cos(q[qi] - spinA), (int)(2 * HAM_BETA)) * sin(q[qi] - spinA);
            bool bugz = false;
            if (l > 0)
            {
                if ((q[qi] < HAM_PI * 0.5) && (q[qi] > -HAM_PI * 0.5))
                {
                    bugz = true;
                    derivative *= -1.0;
                }
                else
                {
                    bugz = true;
                    derivative *= -1.0;
                }
            }
            else
            {
                if ((q[qi] < HAM_PI * 0.5) && (q[qi] > -HAM_PI * 0.5))
                {
                    bugz = true;
                    derivative *= -1.0;
                }
                else
                {
                    bugz = true;
                    derivative *= -1.0;
                }
            }
            assert(!isnan(derivative));
            atomicAdd(dHdq + qi, derivative * (angle_t)0.005);
        }
        // test
        int l = c_lits[lidx];
        int qi = abs(l) - 1;
        // angle_t derivative = HAM_2BTA * rval;
        angle_t spinA = HAM_PI * 0.5;
        if (l < 0)
        {
            spinA = HAM_PI * 1.5;
        }
        angle_t derivative = ((angle_t)HAM_2BTA) * pow(cos(q[qi] - spinA), (int)(2 * HAM_BETA)) * sin(q[qi] - spinA);
        bool bugz = false;
        if (l > 0)
        {
            if ((q[qi] < HAM_PI * 0.5) && (q[qi] > -HAM_PI * 0.5))
            {
                bugz = true;
                derivative *= -1.0;
            }
            else
            {
                bugz = true;
                derivative *= -1.0;
            }
        }
        else
        {
            if ((q[qi] < HAM_PI * 0.5) && (q[qi] > -HAM_PI * 0.5))
            {
                bugz = true;
                derivative *= -1.0;
            }
            else
            {
                bugz = true;
                derivative *= -1.0;
            }
        }
        // if(bugz){
        //     printf("l: %d q: %f d: %f\n",l,q[qi], derivative);
        //     assert(!bugz);
        // }
        // derivative *= tan(q[qi] - spinA);
        // printf("[%d] += %f\n",qi, derivative);
        atomicAdd(dHdq + qi, derivative);
    }

    angle_t calc_avg(angle_t *d_data, int n, void *&tmpd, size_t &tmps)
    {
        angle_t *av;
        cudaMalloc(&av, sizeof(angle_t));
        if (tmpd == NULL)
        {
            cub::DeviceReduce::Sum(tmpd, tmps, d_data, av, n);
            ccE(cudaMalloc(&tmpd, tmps));
        }
        cub::DeviceReduce::Sum(tmpd, tmps, d_data, av, n);
        angle_t h_av = 0.0;
        cudaMemcpy(&h_av, av, sizeof(angle_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        h_av = h_av / ((angle_t)n);
        return h_av;
    }

    void terminate(Q_data &Q)
    {
        g_lock.lock();
        printf("TERMINATED! LCNT: %ld COST: %d\n", Q.loops, Q.cost);
        g_lock.unlock();
        printV(Q);
    }

    void ls(Q_data &Q)
    {
        cub::KeyValuePair<int, angle_t> *d_maxlit;
        ccE(cudaMallocAsync(&d_maxlit, sizeof(cub::KeyValuePair<int, angle_t>) * h_ln, Q.s));
        std::chrono::time_point<std::chrono::steady_clock> t1 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t1 - t0;
        size_t tsb_lv = 0;
        void *tmps_lv = NULL;
        size_t tsb_st = 0;
        void *tmps_st = NULL;
        size_t tsb_av = 0;
        void *tmps_av = NULL;
        size_t tsb_av2 = 0;
        void *tmps_av2 = NULL;
        bool timeout = false;
        angle_t avgp, avgdh;
        uint32_t best = h_cn;
        uint32_t tuc = 0;
        while (Q.cost > 0 & !timeout)
        {
            Q.loops++;
            update_lV<<<bszL, tszL, 0, Q.s>>>(Q._r, Q._lV);
            if (tmps_lv == NULL)
            {
                cub::DeviceSegmentedReduce::ArgMax(tmps_lv, tsb_lv, Q._lV, d_maxlit, h_cn, d_indxs, d_indxs + 1, Q.s);
                ccE(cudaMallocAsync(&tmps_lv, tsb_lv, Q.s));
            }
            cub::DeviceSegmentedReduce::ArgMax(tmps_lv, tsb_lv, Q._lV, d_maxlit, h_cn, d_indxs, d_indxs + 1, Q.s);
            // ccE(cudaDeviceSynchronize());
            ccE(cudaMemsetAsync(Q._dHdq, 0, sizeof(angle_t) * h_vn, Q.s));
            calc_dHdq<<<bszC, tszC, 0, Q.s>>>(d_maxlit, Q._q, Q._r, Q._dHdq, d_indxs);
            // cudaDeviceSynchronize();
            // pN<<<1,1>>>(Q._dHdq, h_vn);
            // cudaDeviceSynchronize();
            // exit(0);
            // d
            // cudaDeviceSynchronize();
            // pN<<<1,1>>>(Q._dHdq, 100);
            // cudaDeviceSynchronize();
            // d
            angle_t df = 1.0;
            df = 0.99999;
            update_p<<<bszV, tszV, 0, Q.s>>>(Q._p, Q._dHdq, df);
            // cudaDeviceSynchronize();
            // g_lock.lock();
            // printf("P: ");
            // pN<<<1,1>>>(Q._p, 5);
            // // exit(0);
            // cudaDeviceSynchronize();
            // printf("Q: ");
            // pN<<<1,1>>>(Q._q, 5);
            // cudaDeviceSynchronize();
            // g_lock.unlock();
            update_qr<<<bszV, tszV, 0, Q.s>>>(Q._q, Q._r, Q._p);
            // ccE(cudaStreamSynchronize(Q.s));
            discretize<<<bszV, tszV, 0, Q.s>>>(Q._q, Q.dvs);
            calc_satf<<<bszL, tszL, 0, Q.s>>>(Q.dvs, Q.satf);
            uint32_t cost = calc_cost(Q, tmps_st, tsb_st);
            if(cost < best){
                tuc = 0;
                best = cost;
            }else{
                tuc++;
            }
            if( tuc > 10000){
                // jolt
                tuc = 0;
                printf("####\n###\n##\nJOLT\n##\n###\n####\n");
                initVQP<CQF><<<bszI, tszI, 0, Q.s>>>(Q._p, Q.cst, PJ*100.0);
            }
            avgp = calc_avg(Q._p, h_vn, tmps_av, tsb_av);
            avgdh = calc_avg(Q._dHdq, h_vn, tmps_av2, tsb_av2);
            printf("C: %d | <P>: %f | <dHdq>: %f [%ld]\n", cost, avgp, avgdh, Q.loops);
            // exit(0);
            t1 = std::chrono::steady_clock::now();
            elapsed = t1 - t0;
            if (elapsed.count() > TIMEOUT)
            {
                timeout = true;
                break;
            }
        }
        terminate(Q);
        // if (!timeout)
        // terminate(Q);
        // return; // redundant as terminate exit(0)s
    }

    uint32_t solve(bool ires, bool *&res)
    {
        dprint("solving...\n");
        t0 = std::chrono::steady_clock::now();
        ccE(cudaDeviceSynchronize());
        // pN<<<1,1>>>(d_indxs+5000,20);
        // cudaDeviceSynchronize();
        // exit(0);
        initL2C<<<bszC, tszC, 0, Qs[0].s>>>(d_indxs); // T ~ 20 (us)
        // cudaDeviceSynchronize();
        // pN<<<1,1>>>(d_L2C + 1000, 20);
        // cudaDeviceSynchronize();
        // exit(0);
        // cudaFreeAsync(d_indxs, Qs[1].s);
        cudaStreamSynchronize(Qs[0].s);
        std::thread ts[QS];

#pragma unroll(QS)
        for (auto i = 0; i < QS; i++)
        {
            if (i != 0)
                continue; // for debug
            ts[i] = std::thread(ls, std::ref(Qs[i]));
        }

#pragma unroll(QS)
        for (auto i = 0; i < QS; i++)
        {
            ts[i].join();
        }
    }
}