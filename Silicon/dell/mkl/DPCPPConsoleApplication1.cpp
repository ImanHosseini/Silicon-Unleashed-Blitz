#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "mkl.h"
#include <chrono>
#include <iostream>

int main()
{
    VSLConvTaskPtr task;

    MKL_INT f_shape[] = { 128, 128 };
    MKL_INT g_shape[] = { 3, 3 };
    MKL_INT Rmin[] = { 0, 0 };
    MKL_INT Rmax[] = { f_shape[0] + g_shape[0] - 1, f_shape[1] + g_shape[1] - 1 };
    MKL_INT h_shape[] = { Rmax[0], Rmax[1] };
    MKL_INT h_start[] = { 0, 0 };


    MKL_INT f_stride[] = { f_shape[1], 1 };
    MKL_INT g_stride[] = { g_shape[1], 1 };
    MKL_INT h_stride[] = { h_shape[1], 1 };

    float* f = new float[f_stride[0] * f_shape[0]];
    float* g = new float[g_stride[0] * g_shape[0]];
    float* h = new float[h_stride[0] * h_shape[0]];

    for (int i = 0; i < f_shape[0]; ++i)
        for (int j = 0; j < f_shape[1]; ++j)
            f[i * f_stride[0] + j] = 10;

    for (int i = 0; i < g_shape[0]; ++i)
        for (int j = 0; j < g_shape[1]; ++j)
            g[i * g_stride[0] + j] = 10;

    memset(h, 0, sizeof(h[0]) * h_stride[0] * h_shape[0]);

    int status;
    status = vslsConvNewTask(&task, VSL_CONV_MODE_AUTO, 2, f_shape, g_shape, h_shape);
    assert(status == VSL_STATUS_OK);

    status = vslConvSetStart(task, h_start);
    assert(status == VSL_STATUS_OK);

    auto t1 = std::chrono::high_resolution_clock::now();
    status = vslsConvExec(task, f, f_stride, g, g_stride, h, h_stride);
    assert(status == VSL_STATUS_OK);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "exec time: " << ms_int.count() << "us\n";
    status = vslConvDeleteTask(&task);
    assert(status == VSL_STATUS_OK);

    for (int i = 0; i < h_shape[0]; ++i)
    {
        printf("%3i: ", i);
        for (int j = 0; j < h_shape[1]; ++j)
        {
            printf("%4.0f ", h[i * h_stride[0] + j]);
        }
        printf("\n");
    }
    return 0;
}
