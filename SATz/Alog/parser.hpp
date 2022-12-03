#ifndef _parser_hpp_INCLUDED
#define _parser_hpp_INCLUDED

/*
    based on how CaDiCaL parses
*/

#include <cstdio>
#include <stdio.h>
#include <cstdlib>

#define NS 4096

namespace Melkor{
    class Parser{
        public:
        FILE * file;
        int clauses = 0; 
        int varnum = 0;
        int lit_c = 0;
        void parse_dimacs(int* &lits, int* &indxs);
        void parse_cv();
        void parse_lit(int& ch, int& res);
        void litadd(int* &lits, int lit);
        int parse_char();
        void parse_positive_int(int& ch, int& res);
        Parser(FILE * f) : file(f){};
        private:
        int lno = 0;
        int nlit = 0;
    };
}
#endif