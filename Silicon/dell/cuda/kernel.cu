
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "data.h"

cudaError_t corr(const float *ker, const int kw, const int kh, const float *img, const int iw, const int ih,float* out);

__global__ void corrKer(const float* ker, const int kw, const int kh, const float* img, const int iw, const int ih, float* out)
{
    // printf("IN");
    int bi = blockIdx.x;
    int bj = blockIdx.y;
    int ti = threadIdx.x;
    int tj = threadIdx.y;
    int dbx = IW / gridDim.x;
    int dby = IH / gridDim.y;
    int dtx = dbx / blockDim.x;
    int dty = dby / blockDim.y;
    if (dbx % blockDim.x != 0) {
        dtx += 1;
    }
    if (dby % blockDim.y != 0) {
        dty += 1;
    }
    int xbase = bi * dbx + ti*dtx;
    int ybase = bj * dby + tj*dty;
    for (int x = 0; x < dtx; x++) {
        for (int y = 0; y < dty; y++) {
            if (ti * dtx + x >= dbx || tj * dty + y >= dby) {
                continue;
            }
            int xr = xbase + x;
            int yr = ybase + y;
            if (xr > 609 && yr>458) {
                printf("%d,%d\n", xr, yr);
            }
           
            for (int ki = 0; ki < kw; ki++) {
                for (int kj = 0; kj < kh; kj++) {
                    int dxi = ki - kw / 2;
                    int dxj = kj - kh / 2;
                    out[xr + yr * iw] += img[(xr+dxi+(kw-1)/2)+(yr+dxj+(kh-1)/2)*(iw+kw-1)] * ker[ki+kj*kw];
                }
            }
        }
    }
}

__global__ void idxKer() {
    int bi = blockIdx.x;
    int bj = blockIdx.y;
    int ti = threadIdx.x;
    int tj = threadIdx.y;
    printf("%d, %d\n", ti, tj);
}


int main()
{
   
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    
    // Add vectors in parallel.
    float out[IW*IH];
    cudaError_t cudaStatus = corr(ker,KW,KH,img,IW,IH,out);
    if (cudaStatus != cudaSuccess) {
        printf("FAILED\n");
    }
    FILE* fp;
    fp = fopen("output.txt","w");
    for (int i = 0; i < IH; i++) {
        for (int j = 0; j < IW; j++) {
            fprintf(fp,"%f,",out[i*IW+j]);
        }
        fprintf(fp, "\n");
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t corr(const float* ker, const int kw, const int kh, const float* img, const int iw, const int ih, float* out)
{
    float *dev_ker = 0;
    float *dev_img = 0;
    float *dev_out = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    constexpr int dx = (KW - 1) / 2;
    constexpr int dy = (KH - 1) / 2;
    float pad_img[(IW + 2 * dx) * (IH + 2 * dy)] = {};
    for (int ix = 0; ix < IW; ix++) {
        for (int iy = 0; iy < IH; iy++) {
            pad_img[(iy + dy) * (KW + 2 * dx) + ix + dx] = img[iy * IW + ix];
        }
    }
    cudaStatus = cudaMalloc((void**)&dev_out, IW*IH * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_ker, IW * IH * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_img, (IW+2*dx) * (IH+2*dy) * sizeof(float));
    cudaStatus = cudaMemcpy(dev_ker, ker, KW*KH * sizeof(float), cudaMemcpyHostToDevice);
 
    cudaStatus = cudaMemcpy(dev_img, pad_img, (IW + 2 * dx) * (IH + 2 * dy) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    //cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMalloc failed!");
    //    goto Error;
    //}

    //cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMalloc failed!");
    //    goto Error;
    //}

    //// Copy input vectors from host memory to GPU buffers.
    //cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

    //cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

    // Launch a kernel on the GPU with one thread for each element.
    dim3 dimGrid(4, 4);
    dim3 threads(64, 8);
    auto t1 = std::chrono::high_resolution_clock::now();
    corrKer<<<dimGrid, threads>>>(dev_ker,KW,KH,dev_img,IW,IH,dev_out);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "exec time: " << ms_int.count() << "us\n";
    cudaStatus = cudaMemcpy(out, dev_out, IW*IH * sizeof(float), cudaMemcpyDeviceToHost);
    
    return cudaStatus;
}
