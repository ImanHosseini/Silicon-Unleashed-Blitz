#include "melkor.hpp"

#include <cstdlib>
#include <cstring>

namespace Melkor{

        void printvals(const bool *vals, size_t n)
    {
        for (int i = 0; i < n; i++)
        {
            printf("%d", vals[i]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv){
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*1024*512);
    // FILE * f = fopen("./t0.dimacs","r");
    // char* fname = "/home/iman/projs/MK/Melkor/SatBenchs21/MVD_ADS_S10_6_6.cnf";
    // char* fname = "/scratch/shz230/MK/SatBenchs21/mp1-squ_ali_s10x10_c39_abio_SAT.cnf";
    char* fname = "/home/iman/projs/MK/Melkor/SatBenchs21/mp1-squ_ali_s10x10_c39_abio_SAT.cnf";
    if(argc == 2){
        fname = argv[1];
    }
    FILE * f = fopen(fname,"r");
    if (f == NULL) {
    perror("failed to open file");
    return 1;
}
    Melkor::initST();
    Melkor::Parser* parser = new Melkor::Parser(f);
    parser->parse_cv();
    Melkor::h_cn = parser->clauses;
    Melkor::h_vn = parser->varnum;
    Melkor::initCM0();
    parser->parse_dimacs(Melkor::lits,Melkor::indxs);
    dprint("parsed\n");
    Melkor::h_ln = parser->lit_c;
    Melkor::initCM();
    bool* res = NULL;
    Melkor::solve(true,res);
    // Melkor::printvals(res,varnum);
}