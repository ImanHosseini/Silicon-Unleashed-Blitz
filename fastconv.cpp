// fastconv.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <immintrin.h>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

#define ALIGNMENT 32
#define SIMD_ALIGN __declspec(align(ALIGNMENT))

float sum_ymm(__m256 x) {
	__m128 hi = _mm256_extractf128_ps(x, 1);
	__m128 lo = _mm256_extractf128_ps(x, 0);
	lo = _mm_add_ps(hi, lo);
	hi = _mm_movehl_ps(hi, lo);
	lo = _mm_add_ps(hi, lo);
	hi = _mm_shuffle_ps(lo, lo, 1);
	lo = _mm_add_ss(hi, lo);
	return _mm_cvtss_f32(lo);
}

/*
__m256 _mm256_fmadd_ps (__m256 a, __m256 b, __m256 c)
to lead into __m256 => __m256 t; t = _mm_load_ps(a);
__m128i values = _mm_setr_epi32(0x1234, 0x2345, 0x3456, 0x4567);
int second_value = _mm_extract_epi32(values, 1);
or:
int arrayB[4] = {10, 20, 30, 40};
values = _mm_loadu_si128((__m128i*) arrayB);
*/

#define W 128
#define H 128
#define S 130

void conv2d_d(const float* img, const float* kernel, float* result) {
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			int itop = i * S + j;
			int i_img[9]{ itop, itop + 1,itop + 2,itop + S,itop + S + 1,itop + S + 2,itop + 2 * S,itop + 2 * S + 1,itop + 2 * S + 2 };
			float v = 0.0f;
			for (int k = 0; k < 9; k++) {
				v += img[i_img[k]] * kernel[k];
			}
			result[i * W + j] = v;
		}
	}
}

// Method 2 but using gather
void conv2d_3(const float* img, const float* kernel, float* result) {
	int idx_top = 0;
	int idx_res = 0;
	__m256 krnl = _mm256_load_ps(kernel);
	float* a8s = new float[H * W];
	__m256i index = _mm256_set_epi32(0, 1, 2, S, S + 1, S + 2, 2 * S, 2 * S + 1);
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			idx_top = i * S + j;
			idx_res = i * W + j;
			/* how about using gather? __mm256_i32_gather_ps(base, _m256i videx, int scale)*/
			__m256 px18 = _mm256_i32gather_ps(&img[idx_top], index, 4);
			__m256 v = _mm256_mul_ps(px18, krnl);
			a8s[idx_res] = sum_ymm(v);
		}
	}
	__m256 k9 = _mm256_set1_ps(kernel[8]);
	for (int i = 0; i < H * W / 8; i++) {
		int dh = (i * 8) / W;
		int dx = (i * 8) % W;
		int p9i = (dh + 2) * S + dx + 2;;
		__m256 p9s = _mm256_load_ps(&img[p9i]);
		__m256 v = _mm256_load_ps(&img[8 * i]);
		__m256 res = _mm256_fmadd_ps(p9s, k9, v);
		_mm256_storeu_ps(&result[8 * i], res);
	}
}

// Method 2: break each pixel as 8+1 add the final px9*k9 in another loop
void conv2d_2(const float* img, const float* kernel, float* result) {
	int idx_top = 0;
	int idx_res = 0;
	__m256 krnl = _mm256_load_ps(kernel);
	float* a8s = new float[H * W];
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			idx_top = i * S + j;
			idx_res = i * W + j;
			/* how about using gather? __mm256_i32_gather_ps(base, _m256 videx, int scale)*/
			__m256 px18 = _mm256_set_ps(img[idx_top], img[idx_top + 1], img[idx_top + 2], img[idx_top + S], img[idx_top + S + 1],
				img[idx_top + S + 2], img[idx_top + 2 * S], img[idx_top + 2 * S + 1]);
			__m256 v = _mm256_mul_ps(px18, krnl);
			a8s[idx_res] = sum_ymm(v);
		}
	}
	__m256 k9 = _mm256_set1_ps(kernel[8]);
	for (int i = 0; i < H * W / 8; i++) {
		int dh = (i * 8) / W;
		int dx = (i * 8) % W;
		int p9i = (dh + 2) * S + dx + 2;;
		__m256 p9s = _mm256_load_ps(&img[p9i]);
		__m256 v = _mm256_load_ps(&img[8 * i]);
		__m256 res = _mm256_fmadd_ps(p9s, k9, v);
		_mm256_storeu_ps(&result[8 * i], res);
	}
}
// Method 1: go parallel through (output)-pixel decomposition
void conv2d_1(const float* img, const float* kernel, float* result) {
	int idx_top = 0;
	int idx_res = 0;
	__m256 k1 = _mm256_set1_ps(kernel[0]);
	__m256 k2 = _mm256_set1_ps(kernel[1]);
	__m256 k3 = _mm256_set1_ps(kernel[2]);
	__m256 k4 = _mm256_set1_ps(kernel[3]);
	__m256 k5 = _mm256_set1_ps(kernel[4]);
	__m256 k6 = _mm256_set1_ps(kernel[5]);
	__m256 k7 = _mm256_set1_ps(kernel[6]);
	__m256 k8 = _mm256_set1_ps(kernel[7]);
	__m256 k9 = _mm256_set1_ps(kernel[8]);
	for (int i = 0; i < H; i++) {
		//_mm_prefetch((const char*)&img[(i + 1) * S], 4);
		for (int j = 0; j < W / 8; j++) {
			idx_top = i * S + j;
			idx_res = i * W + j;
			//__m256 v = _mm256_setzero_ps();
			__m256 px1 = _mm256_load_ps(&img[idx_top]);
			__m256 v1 = _mm256_mul_ps(px1, k1);
			__m256 px2 = _mm256_load_ps(&img[idx_top + 1]);
			//__m256 v2 = _mm256_fmadd_ps(px2, k2, v);
			__m256 v2 = _mm256_mul_ps(px2, k2);
			__m256 px3 = _mm256_load_ps(&img[idx_top + 2]);
			__m256 v3 = _mm256_mul_ps(px3, k3);
			__m256 px4 = _mm256_load_ps(&img[idx_top + S]);
			__m256 v4 = _mm256_mul_ps(px4, k4);
			__m256 px5 = _mm256_load_ps(&img[idx_top + S + 1]);
			__m256 v5 = _mm256_mul_ps(px5, k5);
			__m256 px6 = _mm256_load_ps(&img[idx_top + S + 2]);
			__m256 v6 = _mm256_mul_ps(px6, k6);
			__m256 px7 = _mm256_load_ps(&img[idx_top + 2 * S]);
			__m256 v7 = _mm256_mul_ps(px7, k7);
			__m256 px8 = _mm256_load_ps(&img[idx_top + 2 * S + 1]);
			__m256 v8 = _mm256_mul_ps(px8, k8);
			__m256 px9 = _mm256_load_ps(&img[idx_top + 2 * S + 2]);
			__m256 v9 = _mm256_mul_ps(px9, k9);
			/*     if (j == (W / 8 - 1)) {
					 _mm_prefetch((const char*)&img[(i + 1) * S], 4);
				 }*/
			v1 = _mm256_add_ps(v1, v2);
			v8 = _mm256_add_ps(v8, v9);
			v3 = _mm256_add_ps(v3, v4);
			v5 = _mm256_add_ps(v5, v6);
			v5 = _mm256_add_ps(v5, v7);
			v5 = _mm256_add_ps(v5, v3);
			v5 = _mm256_add_ps(v5, v8);
			v1 = _mm256_add_ps(v1, v5);
			_mm256_storeu_ps(&result[idx_res], v1);
		}
		//_mm_prefetch((const char*)&img[(i + 1) * S], 4);
	}
}

// Gaussian Kernel
float kernel[9]{ 1.0f / 16.0f,2.0f / 16.0f,1.0f / 16.0f, 2.0f / 16.0f,4.0f / 16.0f,2.0f / 16.0f, 1.0f / 16.0f,2.0f / 16.0f,1.0f / 16.0f };

// Load Lena + padd (?)
void load_lena(float* img) {
	std::ifstream lena("LENA_s.txt");
	std::string data;
	std::getline(lena, data);
	std::istringstream ss(data);
	int tok;
	int idx = 0;
	while (ss >> std::hex >> tok) {
		img[idx] = float(tok);
		//std::cout << tok << std::endl;
		if (ss.peek() == ',') ss.ignore();
	}
}

int main()
{
	SIMD_ALIGN float* img = new float[(128 + 2) * (128 + 2)];
	load_lena(img);
	SIMD_ALIGN float* res = new float[128 * 128];
	auto t1 = std::chrono::high_resolution_clock::now();
	for (auto i = 0; i < 1; i++) {
		conv2d_1(img, kernel, res);
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
	std::cout << "exec time: " << ms_int.count() << "us\n";
	std::cout << "Hello World!\n";
}

