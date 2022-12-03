#ifndef _sls_cuh_INCLUDED
#define _sls_cuh_INCLUDED

#ifndef SX
#define SX 1
#endif

#ifndef QS
#define QS 8
#endif

#define CL 0
#define ST 1
#define UT 2

namespace Melkor
{
        extern uint32_t h_vn;
        extern uint32_t h_cn;
        extern uint32_t h_ln;
        extern uint32_t h_rs;
        extern size_t vsize; 
        // extern uint32_t 
        // extern int n_lits;
        extern int *indxs;
        extern int *lits;
        extern int *d_lits;
        extern int *d_indxs;
        uint32_t solve(bool, bool *&);
        void initCM0();
        void initCM();
        void initST();
}

#endif