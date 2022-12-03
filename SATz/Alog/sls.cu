#include "melkor.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <chrono>
#include <cuda_profiler_api.h>
#include <thread>
#include <algorithm>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <numeric>
// #include <stdbool.h> 

#define TIMEOUT 5000 * 1000
#define EPS 10.0
#define DT 0.001
#define AMX 1.0e12



namespace Melkor
{
    std::mutex g_lock;
    std::mutex qb_lock;
    std::condition_variable cv;
    bool BQ_rdy = false;
    bool HV_rdy = false;

    __device__ void AtomicMaxD(double *address, const double value)
    {
        if (*address >= value)
        {
            return;
        }

        unsigned long long int *address_as_i = (unsigned long long int *)address;
        unsigned long long int old = *address_as_i, assumed;

        do
        {
            assumed = old;
            if (__longlong_as_double(assumed) >= value)
            {
                break;
            }

            old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
        } while (assumed != old);
    }

    typedef struct B_data
    {
        uint32_t icost;
        cudaStream_t streamA;
        // |v_p0|v_n0|v_p1|v_n1| ...
        uint32_t* d_val_ofs;
        // |v0_c0+|v0_c1+|v0_c2-|...|v0_cn| ...
        // ^_____________^____...
        uint32_t* d_val_cls;
        uint32_t* d_htbl;
        bool* d_fusev;
        uint8_t* d_basesat;
        double* d_sdot;
        bool* d_vals;
        // uint32_t h_vids[BFACTOR];
    } B_data;
 

    void prep_vtbls(B_data& B){
        ccE(cudaMallocAsync(&(B.d_val_cls),sizeof(uint32_t)*h_ln, B.streamA));
        ccE(cudaMallocAsync(&(B.d_val_ofs),sizeof(uint32_t)*(2*h_vn+1),B.streamA));
        uint32_t* h_val_cls = new uint32_t[h_ln];
        uint32_t* h_val_ofs = new uint32_t[2*h_vn + 1];
        std::vector<std::vector<int>> v_pa(h_vn, std::vector<int>());
        std::vector<std::vector<int>> v_pn(h_vn, std::vector<int>());
        size_t data_c = 0;
        for(int cli = 0; cli < h_cn; cli++){
            int i0 = indxs[cli];
            int i1 = indxs[cli+1];
            for(int lidx = i0; lidx < i1; lidx++){
                int l = lits[lidx];
                if( l > 0){
                    v_pa[l-1].push_back(cli);
                }else{
                    v_pn[abs(l)-1].push_back(cli);
                }
            }
        }
        int cur = 0;
        for(int vi = 0; vi < h_vn; vi++){
            int p_cnt = v_pa[vi].size();
            int n_cnt = v_pn[vi].size();
            h_val_ofs[2*vi] = cur;
            h_val_ofs[2*vi + 1] = cur + p_cnt;
            for(int ci = 0; ci < p_cnt; ci++){
                h_val_cls[cur + ci] = v_pa[vi][ci];
            }
            for(int ci = 0; ci < n_cnt; ci++){
                h_val_cls[cur + p_cnt + ci] = v_pn[vi][ci];
            }
            cur += p_cnt + n_cnt;
        }
        h_val_ofs[2*h_vn] = cur;
        ccE(cudaMemcpyAsync(B.d_val_ofs, h_val_ofs, sizeof(uint32_t)*(2*h_vn+1), cudaMemcpyHostToDevice, B.streamA));
        ccE(cudaMemcpyAsync(B.d_val_cls, h_val_cls, sizeof(uint32_t)*h_ln, cudaMemcpyHostToDevice, B.streamA));
    }

    typedef struct Q_data
    {
        int Qnumber;
        cudaStream_t streamA;
        cudaStream_t streamB;
        double *sA;
        double *sB;
        // double* sB;
        double *K;
        double *aA;
        double *aB;
        double *Z;
        double *sdot;
        bool *vals;
        uint8_t *satf;
        uint32_t *d_sat;
        uint32_t satcnt = 0;
        /*
         hN
         layout: [h0|h0*|h1|h1*|...]
         where * is for flipped vals
         NOW: REUSE hS as hN TOO!
        */
        // uint32_t *hN = NULL;
        // uint32_t *flg = NULL;
        uint32_t *flgS = NULL;
        curandState_t *cst;
        uint32_t loops = 0;
        // Q_data() : s(NULL), valQ(NULL), valM(NULL), hMax(0), hQ(0), hN(NULL), bVS(NULL), d_argmin(NULL){}
    } Q_data;

    Q_data Q0;
    B_data B0;
    // void launch_spec(int &bsz, int &tsz, int n)
    // {
    //     bsz = (n + 511) / 512;
    //     tsz = 512;
    // }

    // These are 'extern'ed in .cuh
    uint32_t h_vn;
    uint32_t max_s;
    uint32_t h_cn;
    uint32_t h_ln;
    uint32_t h_rs;
    int vsize;
    int fsize;
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

    dim3 bszL;
    dim3 tszL;
    dim3 bszS;
    dim3 tszS;
    dim3 bszC;
    dim3 tszC;
    uint32_t bszI;
    uint32_t tszI;
    uint32_t bszV;
    uint32_t tszV;
    __constant__ int *c_lits;
    int *d_L2C;
    __constant__ int *c_L2C;

    // utility func for debug printing gpu memory
    // BEWARE: device printf cannot handle too much data! (or resize printf buffer)
    template <typename T>
    __global__ void pN(const T *p, int n)
    {
        printf("[");
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

    template <int S>
    __global__ void initVQS(double *v, curandState_t __restrict__ *cst)
    {
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = 0; i < S; i++)
        {
            int j = tidx + i * stride;
            if (j >= c_vn)
                return;
            double v0 = curand_uniform_double(cst + tidx);
            v[j] = (v0 * 2.0) - 1.0;
        }
    }

    inline int kl_init(uint32_t &bsz, uint32_t &tsz, int wpt = 1)
    {
        int k = (h_vn + wpt - 1) / wpt;
        bsz = (k + 511) / 512;
        tsz = (k + bsz - 1) / bsz;
        dprint("kernel<%d><<<%d,%d>>>\n", wpt, bsz, tsz);
        return k;
    }

    inline void kl_i(dim3 &bsz, dim3 &tsz, uint32_t items)
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

    void printV(const uint8_t *__restrict__ vals, bool flipped = false)
    {
        printf("V:");
        for (int i = 0; i < vsize; i++)
        {
            printf("%d", ((vals[i] >> 0) & 1) ^ flipped);
            printf("%d", ((vals[i] >> 1) & 1) ^ flipped);
            printf("%d", ((vals[i] >> 2) & 1) ^ flipped);
            printf("%d", ((vals[i] >> 3) & 1) ^ flipped);
            printf("%d", ((vals[i] >> 4) & 1) ^ flipped);
            printf("%d", ((vals[i] >> 5) & 1) ^ flipped);
            printf("%d", ((vals[i] >> 6) & 1) ^ flipped);
            printf("%d", ((vals[i] >> 7) & 1) ^ flipped);
        }
        printf("\n");
    }

    __global__ void calc_htbl_fused(uint32_t __restrict__* htbl,
    const uint32_t __restrict__* val_ofs, const uint32_t __restrict__* val_cls,
    const uint32_t __restrict__* vixs, const uint8_t __restrict__* satfuse){
        // SAT_FUSE IS UNPACKED - 1 uint8_t per clause
        uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t ti = tidx;
        uint8_t* flg = (uint8_t*)malloc(((c_cn + 7) / 8) * sizeof(uint8_t));
        assert(flg != NULL);
        memset(flg, 0x0, sizeof(uint8_t)*(c_cn + 7) / 8);
        for(int i = 0; i < BFACTOR; i++){
            int vidx = vixs[i];
            bool vi = ti % 2;
            ti = ti >> 1;
            int j0, j1;
            if(vi){
                j0 = val_ofs[2*vidx];
                j1 = val_ofs[2*vidx+1];
            }else{
                j0 = val_ofs[2*vidx+1];
                j1 = val_ofs[2*vidx+2];
            }
            for(int j = j0; j < j1; j++){
                int cl = val_cls[j];
                uint8_t bm = 1 << (cl % 8);
                if(satfuse[cl]) continue;
                uint8_t flgi = flg[cl / 8];
                if(flgi & bm){

                }else{
                    flg[cl / 8] |= bm;
                    htbl[tidx] += 1;
                }
            }
        }
        free(flg);
    }

        // DEPRECATED! NO FUSE! DON'T USE!
       __global__ void calc_htbl(uint32_t __restrict__* htbl, const uint32_t __restrict__* val_ofs, const uint32_t __restrict__* val_cls, const uint32_t __restrict__* vixs){
        uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t ti = tidx;
        uint8_t* flg = (uint8_t*)malloc(((c_cn + 7) / 8) * sizeof(uint8_t));
        assert(flg != NULL);
        for(int i = 0; i < BFACTOR; i++){
            int vidx = vixs[i];
            bool vi = ti % 2;
            ti = ti >> 1;
            int j0, j1;
            if(vi){
                j0 = val_ofs[2*vidx];
                j1 = val_ofs[2*vidx+1];
            }else{
                j0 = val_ofs[2*vidx+1];
                j1 = val_ofs[2*vidx+2];
            }
            for(int j = j0; j < j1; j++){
                int cl = val_cls[j];
                uint8_t flgi = flg[cl / 8];
                uint8_t bm = 1 << (cl%8);
                if(flgi & bm){

                }else{
                    flg[cl / 8] += bm;
                    htbl[tidx] += 1;
                }
            }
        }
        free(flg);
    }

    __global__ void calc_satf(const bool *__restrict__ vals, __restrict__ uint8_t *satf)
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

        __global__ void calc_satf_fused(const bool *__restrict__ vals, __restrict__ uint8_t *satf, const bool *__restrict__ fuse)
    {
        uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
        uint32_t tidx = bid * blockDim.x + threadIdx.x;
        if (tidx >= c_ln)
        {
            return;
        }
        int l = c_lits[tidx];
        int ri = abs(l) - 1;
        if(fuse[ri]){
            return;
        }
        bool rv = vals[ri];
        bool sat = !(rv ^ (l > 0));
        if (sat)
        {
            int cli = c_L2C[tidx];
            satf[cli] = 1;
        }
    }

    __global__ void update_sdot(const double *__restrict__ a, const double *__restrict__ Z, const double *__restrict__ K, double *__restrict__ sdot)
    {
        int bid = blockIdx.y * gridDim.x + blockIdx.x;
        int tid = bid * blockDim.x + threadIdx.x;
        if (tid >= c_ln)
            return;
        int l = c_lits[tid];
        int i = abs(l) - 1;
        int cli = c_L2C[tid];
        // printf("%d\n",cli);
        double km = K[cli];
        double v = 2.0 * a[cli] * km;
        if (l < 0)
            v *= -1.0;
        // RISK OF NUMERICAL ISSUES!
        double divi = -1.337;
        if (km != 0.0)
        {
            divi = Z[tid];
            if (divi == 0.0)
            {
                printf("km:%f am:%f\n", km, a[cli]);
                assert(false);
            }
            v *= km / divi;
        }
        // printf("A:%d:%f\n",i, v);
        atomicAdd(sdot + i, v);
        if (isnan(v))
        {
            printf("km:%f am:%f divi:%f\n", km, a[cli], divi);
        }
        assert(!isnan(v));
    }

    __global__ void update_s(const double *__restrict__ sdot, const double *__restrict__ s, double *__restrict__ snew, double dt, double *smx)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= c_vn)
            return;
        // maybe check S in range [-1, 1] ?
        double sdot_i = sdot[tid];
        AtomicMaxD(smx, abs(sdot_i));
        double val = fma(dt, sdot_i, s[tid]);
        if (val > 1.0)
        {
            val = saturate(val);
        }
        else if (val < -1.0)
        {
            val = -saturate(-val);
        }
        snew[tid] = val;
    }

    __global__ void stats(const double *__restrict__ vs, int cnt, double vmin, double vmax, double avg)
    {
        double v_min = 100000.0;
        double v_max = -100000.0;
        double sum = 0.0;
        for (int i = 0; i < cnt; i++)
        {
            double val = vs[i];
            if (val < v_min)
                v_min = val;
            if (val > v_max)
                v_max = val;
            sum += val;
        }
        avg = sum / (double)cnt;
        vmin = v_min;
        vmax = v_max;
        printf("[MIN: %f| MAX: %f| AVG: %f]\n", vmin, vmax, avg);
    }

    __global__ void stats(const double *__restrict__ vs, int cnt)
    {
        double v_min = 100000.0;
        double v_max = -100000.0;
        double sum = 0.0;
        for (int i = 0; i < cnt; i++)
        {
            double val = vs[i];
            if (val < v_min)
                v_min = val;
            if (val > v_max)
                v_max = val;
            sum += val;
        }
        sum = sum / (double)cnt;
        printf("[MIN: %.20f| MAX: %.20f| AVG: %.20f]\n", v_min, v_max, sum);
    }

    __global__ void s2v(const double *__restrict__ s, bool *__restrict__ v)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= c_vn)
            return;
        v[tid] = fma(0.5, s[tid], 0.5) > 0.5;
    }

    __global__ void update_a(const double *__restrict__ a, const double *__restrict__ K, double *__restrict__ a_new, double *amx, double dt, const int *__restrict__ indxs)
    {
        int bid = blockIdx.y * gridDim.x + blockIdx.x;
        int tid = bid * blockDim.x + threadIdx.x;
        if (tid >= c_cn)
            return;
        // double dtk2_ = K[tid] * K[tid];
        double dtk2_ = abs(K[tid]);
        // scaling
        double sc = double(1 << (indxs[tid+1] - indxs[tid]));
        dtk2_ /= sc;
        double dtk2 = fma(dt, dtk2_, 1.0);
        dtk2 *= a[tid];
        AtomicMaxD(amx, dtk2);
        a_new[tid] = dtk2;
    }

    __global__ void rescale(double *__restrict__ a, double f)
    {
        int bid = blockIdx.y * gridDim.x + blockIdx.x;
        int tid = bid * blockDim.x + threadIdx.x;
        if (tid >= c_cn)
            return;
        a[tid] = a[tid] * f;
    }

    __global__ void updateZ(const double *__restrict__ s, double *__restrict__ z)
    {
        int bid = blockIdx.y * gridDim.x + blockIdx.x;
        int tid = bid * blockDim.x + threadIdx.x;
        if (tid >= c_ln)
            return;
        int l = c_lits[tid];
        int si = abs(l) - 1;
        double c = 1.0;
        if (l < 0)
            c = -1.0;
        double zval = (1.0 - c * s[si]);
        z[tid] = zval;
    }

    void printV(Q_data &Q)
    {
        bool *v = (bool *)malloc(sizeof(bool) * h_vn);
        ccE(cudaMemcpyAsync(v, Q.vals, sizeof(bool) * h_vn, cudaMemcpyDeviceToHost, Q.streamB));
        ccE(cudaStreamSynchronize(Q.streamB));
        printf("Q%d:\n", Q.Qnumber);
        for (auto i = 0; i < h_vn; i++)
        {
            printf("%d", v[i]);
        }
        printf("\n");
        free(v);
    }

    uint32_t calc_cost(Q_data &Q, void *&tmps, size_t &tsb, double *s)
    {
        s2v<<<bszV, tszV, 0, Q.streamB>>>(s, Q.vals);
        ccE(cudaMemsetAsync(Q.satf, 0, sizeof(uint8_t) * h_cn, Q.streamB));
        calc_satf<<<bszL, tszL, 0, Q.streamB>>>(Q.vals, Q.satf);
        if (tmps == NULL)
        {
            cub::DeviceReduce::Sum(tmps, tsb, Q.satf, Q.d_sat, h_cn, Q.streamB);
            ccE(cudaMallocAsync(&tmps, tsb, Q.streamB));
        }
        cub::DeviceReduce::Sum(tmps, tsb, Q.satf, Q.d_sat, h_cn, Q.streamB);
        ccE(cudaMemcpyAsync(&Q.satcnt, Q.d_sat, sizeof(uint32_t), cudaMemcpyDeviceToHost, Q.streamB));
        ccE(cudaStreamSynchronize(Q.streamB));
        uint32_t cost = h_cn - Q.satcnt;
        Q.satcnt = cost;
        // printV(Q);
        return cost;
    }

    void initST()
    {
        // ccE(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*1024*512));
        ccE(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 620ll * (1ll<<BFACTOR)));
        // cudaDeviceScheduleBlockingSync or:
        /*
        :in ascending order from Low latency to High latency:
        CudaDeviceScheduleSpin,
        CudaDeviceScheduleYield,
        cudaDeviceScheduleBlockingSync
        */
        // ccE(cudaSetDeviceFlags(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)));
        Q0.Qnumber = 0;
        ccE(cudaStreamCreateWithFlags(&(Q0.streamA), cudaStreamNonBlocking));
        ccE(cudaStreamCreateWithFlags(&(Q0.streamB), cudaStreamNonBlocking));
        ccE(cudaStreamCreateWithFlags(&(B0.streamA), cudaStreamNonBlocking));
        ccE(cudaMallocAsync(&Q0.d_sat, sizeof(uint32_t), Q0.streamA));
        ccE(cudaMallocAsync(&B0.d_htbl, sizeof(uint32_t) * (1 << BFACTOR), B0.streamA));
        // ccE(cudaMallocAsync(&(Qs[i].d_argmin), sizeof(cub::KeyValuePair<int, int>), Qs[i].s));
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
        // Q0.satcnt = h_cn;
        int k = kl_init(bszI, tszI, CQF);
        kl_init(bszV, tszV, 1);
        kl_i(bszC, tszC, h_cn);
        ccE(cudaMallocAsync(&(Q0.cst), sizeof(curandState_t) * bszI * tszI, Q0.streamA));
        ccE(cudaMallocAsync(&(Q0.sA), sizeof(double) * h_vn, Q0.streamA));
        ccE(cudaMallocAsync(&(Q0.sB), sizeof(double) * h_vn, Q0.streamA));
        ccE(cudaMallocAsync(&(Q0.vals), sizeof(bool) * h_vn, Q0.streamB));
        ccE(cudaMallocAsync(&(B0.d_vals), sizeof(bool) * h_vn, B0.streamA));
        // ccE(cudaMallocAsync(&(Q0.sB), sizeof(double) * h_vn, Q0.streamA));
        ccE(cudaMallocAsync(&(Q0.sdot), sizeof(double) * h_vn, Q0.streamA));
        ccE(cudaMallocAsync(&(B0.d_sdot), sizeof(double) * h_vn, B0.streamA));
        ccE(cudaMallocAsync(&(Q0.aA), sizeof(double) * h_cn, Q0.streamB));
        ccE(cudaMallocAsync(&(Q0.aB), sizeof(double) * h_cn, Q0.streamB));
        ccE(cudaMallocAsync(&(Q0.K), sizeof(double) * h_cn, Q0.streamB));
        ccE(cudaMallocAsync(&(Q0.satf), sizeof(uint8_t) * h_cn, Q0.streamA));
        initCQ<<<bszI, tszI, 0, Q0.streamA>>>(Q0.cst, get_T64(), 0xffffff, k);
        initVQS<CQF><<<bszI, tszI, 0, Q0.streamA>>>(Q0.sA, Q0.cst);
        ccE(cudaMemcpyToSymbolAsync(c_vn, &h_vn, sizeof(int), 0, cudaMemcpyHostToDevice, Q0.streamB));
        ccE(cudaMemcpyToSymbolAsync(c_cn, &h_cn, sizeof(int), 0, cudaMemcpyHostToDevice, Q0.streamB));
        ccE(cudaMallocAsync(&d_indxs, sizeof(int) * (h_cn + 1), Q0.streamB));
        // ccE(cudaMemcpyToSymbolAsync(d_indxs_C, &d_indxs, sizeof(int *), 0, cudaMemcpyHostToDevice, sA));
    }

    void initCM()
    {
        ccE(cudaMemcpyToSymbolAsync(c_ln, &h_ln, sizeof(int), 0, cudaMemcpyHostToDevice, Q0.streamA));
        kl_i(bszL, tszL, h_ln);
        // LATER: statically tune and optimize these parameters
        ccE(cudaMallocAsync(&d_lits, sizeof(int) * h_ln, Q0.streamA));
        ccE(cudaMallocAsync(&Q0.Z, sizeof(double) * h_ln, Q0.streamA));
        ccE(cudaMemcpyAsync(d_lits, lits, sizeof(int) * h_ln, cudaMemcpyHostToDevice, Q0.streamA));
        ccE(cudaMallocAsync(&d_L2C, sizeof(int) * h_ln, Q0.streamB));
        ccE(cudaMemcpyToSymbolAsync(c_lits, &d_lits, sizeof(int *), 0, cudaMemcpyHostToDevice, Q0.streamA));
        ccE(cudaMemcpyToSymbolAsync(c_L2C, &d_L2C, sizeof(int *), 0, cudaMemcpyHostToDevice, Q0.streamB));
        ccE(cudaMemcpyAsync(d_indxs, indxs, sizeof(int) * (h_cn + 1), cudaMemcpyHostToDevice, Q0.streamB));
    }

    __global__ void initL2Ca(const int *__restrict__ indxs, double *__restrict__ a)
    {
        int bid = blockIdx.y * gridDim.x + blockIdx.x;
        int tid = bid * blockDim.x + threadIdx.x;
        if (tid >= c_cn)
            return;
        a[tid] = 0.0001;
        int i0 = indxs[tid];
        int i1 = indxs[tid + 1];
        for (int i = i0; i < i1; i++)
        {
            c_L2C[i] = tid;
        }
    }

    inline void terminate(Q_data &Q0)
    {
    }

    struct CustomMul
    {
        template <typename T>
        __device__ __forceinline__
            T
            operator()(const T &a, const T &b) const
        {
            return a * b;
        }
    };

    void bforcer(B_data& B){
        // allocate fuse & base
        ccE(cudaMallocAsync(&B.d_fusev,sizeof(bool) * h_vn, B.streamA));
        ccE(cudaMallocAsync(&B.d_basesat,sizeof(uint8_t) * h_cn, B.streamA));
        prep_vtbls(B);
        double* h_sdot = new double[h_vn];
        bool* h_fuse = new bool[h_vn];
        std::vector<uint32_t> hsd_i(h_vn);
        uint32_t* d_vids;
        ccE(cudaMallocAsync(&d_vids, sizeof(uint32_t)*BFACTOR, B.streamA));
        uint32_t base_score = 0;
        uint32_t* d_bscore;
        ccE(cudaMallocAsync(&d_bscore, sizeof(uint32_t), B.streamA));
        void* tmp_data = NULL;
        size_t tmp_data_size;
        void* tmp_data_ht = NULL;
        size_t tmp_data_size_ht;
        cub::KeyValuePair<int, uint32_t> *d_maxh = NULL;
        ccE(cudaMallocAsync(&d_maxh, sizeof(cub::KeyValuePair<int, uint32_t>),B.streamA));
        // MAIN B-LOOP
        while(true){
        {
            std::unique_lock lk(qb_lock);
            BQ_rdy = true;
            cv.wait(lk, []{return HV_rdy;});
            BQ_rdy = false;
        }
        // solve stuff
        ccE(cudaMemcpyAsync(h_sdot,B.d_sdot,h_vn*sizeof(double),cudaMemcpyDeviceToHost, B.streamA));
        ccE(cudaStreamSynchronize(B.streamA));
        // PICK top BFACTOR elements based on sdot value 
        std::iota(hsd_i.begin(), hsd_i.end(), 0);
        std::partial_sort(hsd_i.begin(), hsd_i.begin() + BFACTOR, hsd_i.end(),
        [&h_sdot](uint32_t i1, uint32_t i2){return abs(h_sdot[i2])<abs(h_sdot[i1]);});
       
        ccE(cudaMemcpyAsync(d_vids, hsd_i.data(), sizeof(uint32_t)*BFACTOR, cudaMemcpyHostToDevice, B.streamA));
        memset(h_fuse, 0x0, sizeof(bool)*h_vn);
        ccE(cudaStreamSynchronize(B.streamA));
        for(int i = 0; i < BFACTOR; i++){
            h_fuse[hsd_i[i]] = true;
        }
        ccE(cudaMemsetAsync(B.d_basesat, 0, sizeof(uint8_t) * h_cn, B.streamA));
        ccE(cudaMemcpyAsync(B.d_fusev, h_fuse, sizeof(bool) * h_vn, cudaMemcpyHostToDevice, B.streamA));
        calc_satf_fused<<<bszL, tszL, 0, B.streamA>>>(B.d_vals, B.d_basesat, B.d_fusev);
        if(tmp_data == NULL){
            cub::DeviceReduce::Sum(tmp_data, tmp_data_size, B.d_basesat, d_bscore, h_cn, B.streamA);
            ccE(cudaMallocAsync(&tmp_data, tmp_data_size, B.streamA));
        }
        cub::DeviceReduce::Sum(tmp_data, tmp_data_size, B.d_basesat, d_bscore, h_cn, B.streamA);
        ccE(cudaMemcpyAsync(&base_score, d_bscore, sizeof(uint32_t), cudaMemcpyDeviceToHost, B.streamA));
        ccE(cudaMemsetAsync(B.d_htbl, 0x0, sizeof(uint32_t)* (1 << BFACTOR), B.streamA));
        calc_htbl_fused<<<(1<<(BFACTOR - 9)), 512, 0 ,B.streamA>>>(B.d_htbl, B.d_val_ofs, B.d_val_cls, d_vids, B.d_basesat);
        if(tmp_data_ht == NULL){
            cub::DeviceReduce::ArgMax(tmp_data_ht, tmp_data_size_ht, B.d_htbl, d_maxh, 1<<BFACTOR,B.streamA);
            ccE(cudaMallocAsync(&tmp_data_ht, tmp_data_size_ht, B.streamA));
        }
        cub::DeviceReduce::ArgMax(tmp_data_ht, tmp_data_size_ht, B.d_htbl, d_maxh, 1<<BFACTOR,B.streamA);
        // now we know best pt!
        int bestidx, bestscore;
        ccE(cudaMemcpyAsync(&bestidx, &d_maxh[0].key, sizeof(int), cudaMemcpyDeviceToHost, B.streamA));
        ccE(cudaMemcpyAsync(&bestscore, &d_maxh[0].value, sizeof(uint32_t), cudaMemcpyDeviceToHost, B.streamA));
        ccE(cudaStreamSynchronize(B.streamA));
        printf("base score: %d / %d, best: %d\n", base_score, h_cn, bestscore);
        printf("BFORCER: [%d] -> [%d]\n", B.icost, h_cn-(base_score+bestscore));
        // prepped
        }
    }

    void ls(Q_data &Q)
    {
        double* hsd_d = (double*)malloc(sizeof(double)*h_vn);
        double *amx;
        double *sdotmx;
        double *h_amx = new double[1];
        double *h_sdotmx = new double[1];
        ccE(cudaMallocAsync(&amx, sizeof(double), Q.streamA));
        ccE(cudaMallocAsync(&sdotmx, sizeof(double), Q.streamA));
        std::chrono::time_point<std::chrono::steady_clock> t1 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t1 - t0;
        size_t tmps_size = 0;
        void *tmps = NULL;
        bool timeout = false;
        // double* s = Q.sA;
        double *a = Q.aA;
        double *a_new = Q.aB;
        double *s = Q.sA;
        double *s_new = Q.sB;
        CustomMul cmul;
        void *tmps_cost = NULL;
        size_t tsb_cost;
        double dt = 0.001;
        while (Q.satcnt < h_cn)
        {
            *h_sdotmx = 0.0000000001f;
            ccE(cudaMemcpyAsync(sdotmx, h_sdotmx,sizeof(double),cudaMemcpyHostToDevice,Q.streamA));
            ccE(cudaMemcpyAsync(h_sdotmx, sdotmx, sizeof(double), cudaMemcpyDeviceToHost, Q.streamA));
            ccE(cudaMemsetAsync(amx, 0, sizeof(double), Q.streamB));
            Q.loops++;
            ccE(cudaMemsetAsync(Q.sdot, 0, sizeof(double) * h_vn, Q.streamB));
            updateZ<<<bszL, tszL, 0, Q.streamA>>>(s, Q.Z);
            if (tmps == NULL)
            {
                cub::DeviceSegmentedReduce::Reduce(tmps, tmps_size, Q.Z, Q.K, h_cn, d_indxs, d_indxs + 1, cmul, 1.0, Q.streamA);
                ccE(cudaMallocAsync(&tmps, tmps_size, Q.streamA));
            }
            cub::DeviceSegmentedReduce::Reduce(tmps, tmps_size, Q.Z, Q.K, h_cn, d_indxs, d_indxs + 1, cmul, 1.0, Q.streamA);
            // Now we have Km -> update sdot, adot
            ccE(cudaStreamSynchronize(Q.streamA));
            ccE(cudaStreamSynchronize(Q.streamB));
            update_sdot<<<bszL, tszL, 0, Q.streamA>>>(a, Q.Z, Q.K, Q.sdot);
            update_a<<<bszC, tszC, 0, Q.streamB>>>(a, Q.K, a_new, amx, dt, d_indxs);
            ccE(cudaMemcpyAsync(h_amx, amx, sizeof(double), cudaMemcpyDeviceToHost, Q.streamB));
            update_s<<<bszV, tszV, 0, Q.streamA>>>(Q.sdot, s, s_new, dt, sdotmx);
            ccE(cudaMemcpyAsync(h_sdotmx, sdotmx, sizeof(double), cudaMemcpyDeviceToHost, Q.streamA));
            // calc_cost syncs streamB
            uint32_t cost = calc_cost(Q, tmps_cost, tsb_cost, s);
            if(cost < 20){
                // time to move stuff to B0 if B0 rdy
                {
                    std::lock_guard lk(qb_lock);
                    if(BQ_rdy){
                        // do it
                        ccE(cudaMemcpyAsync(B0.d_sdot,Q.sdot,sizeof(double)*h_vn,cudaMemcpyDeviceToDevice,B0.streamA));
                        ccE(cudaMemcpyAsync(B0.d_vals,Q.vals,sizeof(bool)*h_vn,cudaMemcpyDeviceToDevice,B0.streamA));
                        B0.icost = cost;
                        ccE(cudaStreamSynchronize(B0.streamA));
                    }
                    HV_rdy = true;
                    cv.notify_one();
                }
            }
            if (*h_amx > AMX)
            {
                printf(".");
                rescale<<<bszC, tszC, 0, Q.streamA>>>(a_new, (10.0 / (*h_amx)));
            }
            ccE(cudaStreamSynchronize(Q.streamA));
            dt = 0.09 / (*h_sdotmx);
            printf("[%d] @ [%d] - [%lf]\n",cost, Q.loops, elapsed.count());
            // dbg
            // if (Q.loops % 800 == 0)
            // {
            //     ccE(cudaDeviceSynchronize());
            //     printf("SDOT:");
            //     stats<<<1, 1>>>(Q.sdot, h_vn);
            //     ccE(cudaDeviceSynchronize());
            //     printf("S:");
            //     // pN<<<1,1>>>(s_new + 3595,80);
            //     stats<<<1, 1>>>(s_new, h_vn);
            //     ccE(cudaDeviceSynchronize());
            //     printf("A:");
            //     stats<<<1, 1>>>(a_new, h_cn);
            //     ccE(cudaDeviceSynchronize());
            //     // dbg
            //     printf("C: [%d] dt: [%f]\n", cost, dt);
            // }
            // if( cost < 20){
            //     ccE(cudaMemsetAsync(B0.d_htbl, 0x0, sizeof(uint32_t) * (1 << BFACTOR), B0.streamA));
            //     printf("[B]force @ [%d]\n",cost);
            //     std::vector<uint32_t> hsd_i(h_vn);
            //     std::iota(hsd_i.begin(), hsd_i.end(), 0);
            //     std::partial_sort(hsd_i.begin(), hsd_i.begin() + BFACTOR,[&hsd_d](uint32_t i1, uint32_t i2){return (hsd_d[i1]<hsd_d[i2]);});
            //     //   std::partial_sort(s.begin(), s.begin() + 3, s.end());
            // }
            // swap buffers (kewl!)
            double *tmp = a;
            a = a_new;
            a_new = tmp;
            tmp = s;
            s = s_new;
            s_new = tmp;
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
        initL2Ca<<<bszC, tszC, 0, Q0.streamA>>>(d_indxs, Q0.aA); // T ~ 20 (us)
        // cudaFreeAsync(d_indxs, Qs[1].s);
        cudaStreamSynchronize(Q0.streamA);
        std::thread t_bfcr(bforcer, std::ref(B0));
        std::thread t_asat(ls, std::ref(Q0));
        t_asat.join(); 
        t_bfcr.join();
        // std::thread ts[1];
        // ts[0] = std::thread(ls, std::ref(Q0));
        // ts[0].join();
        // q_round();
        // localsearch();
    }
}