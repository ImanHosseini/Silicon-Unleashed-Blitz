#!/bin/bash
nvcc -dc main.cpp sls.cu parser.cpp
nvcc main.o sls.o parser.o -lcudadevrt -o melkor