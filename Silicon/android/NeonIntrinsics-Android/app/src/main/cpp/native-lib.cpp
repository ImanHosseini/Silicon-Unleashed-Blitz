#include <jni.h>
#include <string>
#include <arm_neon.h>
#include <chrono>

//#include "../../../../../../AppData/Local/Android/Sdk/ndk/20.0.5594570/toolchains/llvm/prebuilt/windows-x86_64/lib64/clang/8.0.7/include/arm_neon.h"



using namespace std;

double usElapsedTime(chrono::system_clock::time_point start) {
    auto end = chrono::system_clock::now();

    return chrono::duration_cast<chrono::microseconds>(end - start).count();
}


short* generateRamp(short startValue, short len) {
    short* ramp = new short[len];

    for(short i = 0; i < len; i++) {
        ramp[i] = startValue + i;
    }

    return ramp;
}



chrono::system_clock::time_point now() {
    return chrono::system_clock::now();
}

void conv2d_c(int width, int height, uint8_t* imageData) {
// find center position of kernel (half of kernel size)
    int kCenterX = 1;
    int kCenterY = 1;
    float out[128][128]{0};
    float kernel[9]{ 1.0f/16.0f,2.0f/16.0f,1.0f/16.0f,2.0f/16.0f,4.0f/16.0f,2.0f/16.0f,1.0f/16.0f,2.0f/16.0f,1.0f/16.0f};
    for(int i=0; i < height; ++i)              // rows
    {
        for(int j=0; j < width; ++j)          // columns
        {
            for(int m=0; m < 3; ++m)     // kernel rows
            {
                int mm = 3 - 1 - m;      // row index of flipped kernel
                for(int n=0; n < 3; ++n) // kernel columns
                {
                    int nn = 3 - 1 - n;  // column index of flipped kernel
                    // index of input signal, used for checking boundary
                    int ii = i + (kCenterY - mm);
                    int jj = j + (kCenterX - nn);
                    // ignore input samples which are out of bound
                    if( ii >= 0 && ii < height && jj >= 0 && jj < width )
                        out[i][j] += (float)imageData[ii*width+jj] * kernel[mm*3+nn];
                }
            }
        }
    }
}

void conv2d(int width, int height, uint8_t* imageData){
    uint8x8_t kernel = {1,2,1,2,4,2,1,2};
    uint8_t kernel_last = 1;
    uint8_t data[]{1,2,1,2,4,2,1,2,1};
    uint8_t* newimageData= new uint8_t[width*height];
    for(int i=0;i<width;i++) // i = rows
    {
        for(int j=0;j<height;j++) // j = columns
        {
            for(int k=0;k<9;k+=3)
            {
                data[k] = imageData[i*width + j];
                data[k+1] = imageData[(i+1)*width + (j+1)];
                data[k+2] = imageData[(i+2)*width + (j+2)];
            }
            uint8x8_t pixel = vld1_u8((const unsigned char*)&data);
            uint8_t pixel_last = data[8];
            uint8x8_t result = vmul_u8(kernel,pixel);
            int sum = 0;
            for(int k=0;k<8;k++)
                sum += result[k];
            sum += pixel_last * kernel_last;
            sum = sum/16;
            sum = sum > 255 ? 255 : sum;
            newimageData[i*width + j]= uint8_t(sum);

        }
    }
}

int dotProduct(short* vector1, short* vector2, short len) {
    int result = 0;

    for(short i = 0; i < len; i++) {
        result += vector1[i] * vector2[i];
    }

    return result;
}

int dotProductNeon(short* vector1, short* vector2, short len) {
    const short transferSize = 4;
    short segments = len / transferSize;

    // 4-element vector of zeros
    int32x4_t partialSumsNeon = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);
    int32x4_t sum4 = vdupq_n_s32(0);


    // Main loop (note that loop index goes through segments). Unroll with 4
    int i = 0;
    for(; i+3 < segments; i+=4) {
        // Preload may help speed up sometimes
        // asm volatile("prfm pldl1keep, [%0, #256]" : :"r"(vector1) :);
        // asm volatile("prfm pldl1keep, [%0, #256]" : :"r"(vector2) :);

        // Load vector elements to registers
        int16x8_t v11 = vld1q_s16(vector1);
        int16x4_t v11_low = vget_low_s16(v11);
        int16x4_t v11_high = vget_high_s16(v11);

        int16x8_t v12 = vld1q_s16(vector2);
        int16x4_t v12_low = vget_low_s16(v12);
        int16x4_t v12_high = vget_high_s16(v12);

        int16x8_t v21 = vld1q_s16(vector1+8);
        int16x4_t v21_low = vget_low_s16(v21);
        int16x4_t v21_high = vget_high_s16(v21);

        int16x8_t v22 = vld1q_s16(vector2+8);
        int16x4_t v22_low = vget_low_s16(v22);
        int16x4_t v22_high = vget_high_s16(v22);

        // Multiply and accumulate: partialSumsNeon += vector1Neon * vector2Neon
        sum1 = vmlal_s16(sum1, v11_low, v12_low);
        sum2 = vmlal_s16(sum2, v11_high, v12_high);
        sum3 = vmlal_s16(sum3, v21_low, v22_low);
        sum4 = vmlal_s16(sum4, v21_high, v22_high);

        vector1 += 16;
        vector2 += 16;
    }
    partialSumsNeon = sum1 + sum2 + sum3 + sum4;

	// Sum up remain parts
    int remain = len % transferSize;
    for(i=0; i<remain; i++) {

        int16x4_t vector1Neon = vld1_s16(vector1);
        int16x4_t vector2Neon = vld1_s16(vector2);
        partialSumsNeon = vmlal_s16(partialSumsNeon, vector1Neon, vector2Neon);

        vector1 += 4;
        vector2 += 4;
    }

    // Store partial sums
    int partialSums[transferSize];
    vst1q_s32(partialSums, partialSumsNeon);

    // Sum up partial sums
    int result = 0;
    for(int i = 0; i < transferSize; i++) {
        result += partialSums[i];
    }

    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_neonintrinsics_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    int lastResult = 0;
    int trials = 1;
    uint8_t* img = new uint8_t[128*128];

    auto start = now();
    for (int i = 0; i < trials; i++) {
        conv2d_c(128,128,img);
    }
    auto elapsedTime = usElapsedTime(start);

    // With NEON intrinsics
    // Invoke dotProductNeon and measure performance
    int lastResultNeon = 0;

    start = now();
    for (int i = 0; i < trials; i++) {
        conv2d(128,128,img);
    }
    auto elapsedTimeNeon = usElapsedTime(start);
    delete[] img;

    // Display results
    std::string resultsString =
            "----==== NO NEON ====----\nResult: " + to_string(lastResult)
            + "\nElapsed time: " + to_string((int) elapsedTime) + " us"
            + "\n\n----==== NEON ====----\n"
            + "Result: " + to_string(lastResultNeon)
            + "\nElapsed time: " + to_string((int) elapsedTimeNeon) + " us"
            ;

    return env->NewStringUTF(resultsString.c_str());
}

//extern "C" JNIEXPORT jstring JNICALL
//Java_com_example_neonintrinsics_MainActivity_stringFromJNI(
//        JNIEnv* env,
//        jobject /* this */) {
//
//    // Ramp length and number of trials
//    const int rampLength = 1024;
//    const int trials = 10000;
//
//    // Generate two input vectors
//    // (0, 1, ..., rampLength - 1)
//    // (100, 101, ..., 100 + rampLength-1)
//    auto ramp1 = generateRamp(0, rampLength);
//    auto ramp2 = generateRamp(100, rampLength);
//
//    // Without NEON intrinsics
//    // Invoke dotProduct and measure performance
//    int lastResult = 0;
//
//    auto start = now();
//    for (int i = 0; i < trials; i++) {
//        lastResult = dotProduct(ramp1, ramp2, rampLength);
//    }
//    auto elapsedTime = msElapsedTime(start);
//
//    // With NEON intrinsics
//    // Invoke dotProductNeon and measure performance
//    int lastResultNeon = 0;
//
//    start = now();
//    for (int i = 0; i < trials; i++) {
//        lastResultNeon = dotProductNeon(ramp1, ramp2, rampLength);
//    }
//    auto elapsedTimeNeon = msElapsedTime(start);
//
//    // Clean up
//    delete ramp1, ramp2;
//
//    // Display results
//    std::string resultsString =
//            "----==== NO NEON ====----\nResult: " + to_string(lastResult)
//            + "\nElapsed time: " + to_string((int) elapsedTime) + " ms"
//            + "\n\n----==== NEON ====----\n"
//            + "Result: " + to_string(lastResultNeon)
//            + "\nElapsed time: " + to_string((int) elapsedTimeNeon) + " ms";
//
//    return env->NewStringUTF(resultsString.c_str());
//}