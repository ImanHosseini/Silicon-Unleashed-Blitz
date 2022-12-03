#!/bin/bash
nvcc -std c++17 -dc --expt-relaxed-constexpr -gencode arch=compute_61,code=sm_61 main.cpp sls.cu parser.cpp
nvcc -arch=sm_61 main.o sls.o parser.o -lcudadevrt -o melkor



nvcc -std c++17 -dc -gencode arch=compute_61,code=sm_61 main.cpp sls.cu parser.cpp
