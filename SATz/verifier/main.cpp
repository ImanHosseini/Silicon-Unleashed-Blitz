// This code sucks, don't judge!

#include "parser.hpp"
#include <cstdlib>
#include <cstring>
#include <vector>

#define BREAK_ON_FIRST false

namespace Melkor
{

    void printvals(const bool *vals, size_t n)
    {
        for (int i = 0; i < n; i++)
        {
            printf("%d: %d\n", i, vals[i]);
        }
        // printf("\n");
    }
}

bool check_lit(int lit, std::vector<bool> &val, bool flipped = false)
{
    if (lit > 0)
        return val[abs(lit) - 1]^flipped;
    return (!val[abs(lit) - 1])^flipped;
}

std::vector<bool> readvals(char *inp)
{
    printf("reading witness...\n");
    std::vector<bool> vals;
    int i = 0;
    char ch = inp[i];
    while (ch != '\0')
    {
        if (ch == '0')
            vals.push_back(false);
        if (ch == '1')
            vals.push_back(true);
        ch = inp[++i];
    }
    for (int i = 0; i < vals.size(); i++)
    {
        if (vals[i])
        {
            printf("|%d: %d\n", i, 1);
        }
        else
        {
            printf("|%d: %d\n", i, 0);
        }
    }
    printf("parsed inp : %d\n", vals.size());
    return vals;
}

bool checkCL(int *lits, int *indxs, int cli, std::vector<bool> &vals , bool flipped=false)
{
    int i0 = indxs[cli];
    int i1 = indxs[cli + 1];
    for (int i = i0; i < i1; i++)
    {
        bool r = check_lit(lits[i], vals, flipped);
        if(cli == 16){
            printf("R:%d at i: %d L: %d\n",r,i,lits[i]);
        }
        if (r)
        {
            return true;
        }
    }
    return false;
}

int main(int argc, char **argv)
{
    // FILE * f = fopen("./t0.dimacs","r");
    char *fname = "/home/iman/projs/MK/Melkor/SatBenchs21/MVD_ADS_S10_6_6.cnf";
    if (argc != 3)
    {
        return -1;
    }
    fname = argv[1];
    char *inp = argv[2];
    std::vector<bool> vals = readvals(inp);
    FILE *f = fopen(fname, "r");
    if (f == NULL)
    {
        perror("failed to open file");
        return 1;
    }
    Melkor::Parser *parser = new Melkor::Parser(f);
    parser->parse_cv();
    int *lits;
    int *indxs;
    parser->parse_dimacs(lits, indxs);
    int clauses = parser->clauses;
    int unsat = 0;
    int f_unsat = 0;
    for (int i = 0; i < clauses; i++)
    {
        bool r = checkCL(lits, indxs, i, vals);
        bool rq = checkCL(lits, indxs, i, vals, true);
        if (!r)
        {
            unsat += 1;
            printf("UNSAT CL:%d\n", i);
            if (BREAK_ON_FIRST)
                return 0;
        }else{
            printf("SAT CL:%d\n", i);
        }
if (!rq)
        {
            f_unsat += 1;
            // printf("UNSAT CL*:%d\n", i);
            if (BREAK_ON_FIRST)
                return 0;
        }
    }
    printf("# of UNSAT: %d | SAT: %d\n", unsat,clauses - unsat);
    printf("* - # of UNSAT: %d | SAT: %d\n", f_unsat,clauses - f_unsat);
    return 0;
}