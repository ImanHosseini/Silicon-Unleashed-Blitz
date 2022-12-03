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
#include <cstdint>

// #define DEBUG

#define ERR(x)                                                       \
    do                                                               \
    {                                                                \
        fprintf(stderr, "[ERR]:%s:%d %s\n", __FILE__, __LINE__, #x); \
        exit(-1);                                                    \
    } while (0)

#include "parser.hpp"

#endif