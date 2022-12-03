#include "melkor.hpp"

namespace Melkor
{
    inline int Parser::parse_char()
    {
        int c = getc_unlocked(file);
        return c;
    };
    inline void Parser::parse_positive_int(int &ch, int &res)
    {
        res = ch - '0';
        while (isdigit(ch = parse_char()))
        {
            int digit = ch - '0';
            res = 10 * res + digit;
        }
    }
    inline void Parser::parse_lit(int &ch, int &res)
    {
        // SKIP LEADING SPACES
        while(ch == ' '){
            ch = parse_char();
        }
        int sign = 1;
        if (ch == '-')
        {
            sign = -1;
            ch = parse_char();
        }
        res = 0;
        parse_positive_int(ch, res);
        res *= sign;
    };
    inline void Parser::litadd(int* &lits, int lit)
    {
        // SOME PROBLEMS HAVE LEADING SPACES! FUCKING FUCK! WHY!??!
        // printf("L:%d\n",lit);
        // if(abs(lit)>varnum){
        //     printf("%d\n",lit);
        //     exit(-1);
        // }
        if (lit_c >= nlit)
        {
            // expand
            double dns = (double)clauses / (double)(lno);
            dns *= (double)lit_c*1.04;
            int expected = (int)dns;
            // printf("-> %d\n", expected * sizeof(int));
            lits = (int *)realloc(lits, expected * sizeof(int));
            // printf("done\n");
            nlit = expected;
        }
        lits[lit_c] = lit;
        lit_c++;
    }
    void Parser::parse_cv(){
        printf("parsing..\n");
        int ch = 0;
        for (;;)
        {
            ch = parse_char();
            if (ch != 'c')
                break;
            while ((ch = parse_char() != '\n'))
            {
                if (ch == EOF)
                    ERR("premature EOF");
            }
        }
        if (ch != 'p')
            ERR("expected p");
        ch = parse_char();
        if (ch != ' ')
            ERR("expected space");
        ch = parse_char();
        if (ch != 'c')
            ERR("expected 'c'");
        ch = parse_char();
        if (ch != 'n')
            ERR("expected 'n'");
        ch = parse_char();
        if (ch != 'f')
            ERR("expected 'f'");
        ch = parse_char();
        if (ch != ' ')
            ERR("expected space");
        ch = parse_char();
        parse_positive_int(ch, varnum);
        clauses = 0;
        ch = parse_char();
        parse_positive_int(ch, clauses);
        printf("varnum: %d | clauses: %d\n", varnum,clauses);
    }

    // CALL AFTER parse_cv()
    void Parser::parse_dimacs(int *&lits, int *&indxs)
    {
        nlit = NS;
        lit_c = 0;
        lits = (int *)malloc(NS * sizeof(int));
        indxs = new int[clauses+1];
        indxs[0] = 0;
        int ch = parse_char();
        for (int i = 0; i < clauses; i++)
        {
            while (1)
            {
                int res = 0;
                parse_lit(ch, res);
                if (res == 0)
                {
                    indxs[i+1] = lit_c;
                    break;
                }
                ch = parse_char();
                litadd(lits, res);
            }
            ch = parse_char();
            lno++;
        }
        lits = (int *)realloc(lits, lit_c * sizeof(int));
    };
}