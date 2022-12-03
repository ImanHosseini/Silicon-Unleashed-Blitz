#ifndef _ham_cuh_INCLUDED
#define _ham_cuh_INCLUDED

/*
 try compiling with -ftz=true
 & -fast_math_math
 : --use_fast_math implies --ftz=true --prec-div=false
 and also uses intrinsics
*/

#ifndef QS
#define QS 2
#endif

// work-per-thread for curand kernel
#ifndef CQF
#define CQF 4
#endif

// // work-per-thread for csat
// #ifndef CSF
// #define CSF 16
// #endif

namespace Melkor
{
        extern uint32_t h_vn;
        extern uint32_t max_s;
        extern uint32_t h_cn;
        extern uint32_t h_ln;
        extern uint32_t h_rs;
        extern int vsize; 
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