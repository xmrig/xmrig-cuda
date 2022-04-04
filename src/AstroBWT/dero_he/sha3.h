/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

namespace AstroBWT_Dero_HE {

constexpr int ROUNDS = 24;

__constant__ static const uint64_t rc[ROUNDS] = {
	0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
	0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
	0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
	0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
	0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
	0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
	0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
	0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__constant__ static const int c[25][2] = {
	{ 1, 2}, { 2, 3}, { 3, 4}, { 4, 0}, { 0, 1},
	{ 6, 7}, { 7, 8}, { 8, 9}, { 9, 5}, { 5, 6},
	{11,12}, {12,13}, {13,14}, {14,10}, {10,11},
	{16,17}, {17,18}, {18,19}, {19,15}, {15,16},
	{21,22}, {22,23}, {23,24}, {24,20}, {20,21}
};

__constant__ static const int ppi[25][2] = {
	{0, 0},  {6, 44},  {12, 43}, {18, 21}, {24, 14}, {3, 28},  {9, 20}, {10, 3}, {16, 45},
	{22, 61}, {1, 1},   {7, 6},   {13, 25}, {19, 8},  {20, 18}, {4, 27}, {5, 36}, {11, 10},
	{17, 15}, {23, 56}, {2, 62},  {8, 55},  {14, 39}, {15, 41}, {21, 2}
};

__device__ __forceinline__ uint64_t R64(uint64_t a, int b, int c) { return (a << b) | (a >> c); }

#define ROUND(k) \
do { \
	C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20]; \
	A[t] ^= C[s + 4] ^ R64(C[s + 1], 1, 63); \
	C[t] = R64(A[at], ro0, ro1); \
	A[t] = (C[t] ^ ((~C[c1]) & C[c2])) ^ (k1 & (k)); \
} while (0)

__global__ void __launch_bounds__(32) sha3(const uint8_t* inputs, uint64_t* hashes)
{
	const uint32_t t = threadIdx.x;

	if (t >= 25) {
		return;
	}

	const uint32_t g = blockIdx.x;
	const uint64_t input_offset = 10240 * 2 * g;
	const uint64_t* input = (const uint64_t*)(inputs + input_offset);

	__shared__ uint64_t A[25];
	__shared__ uint64_t C[25];

	A[t] = 0;

	const uint32_t s = t % 5;
	const int at = ppi[t][0];
	const int ro0 = ppi[t][1];
	const int ro1 = 64 - ro0;
	const int c1 = c[t][0];
	const int c2 = c[t][1];
	const uint64_t k1 = (t == 0) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL;
	const uint64_t k2 = (t < 17) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL;

	const uint32_t input_size = 9973 * 2;
	const uint32_t input_words = input_size / sizeof(uint64_t);
	const uint64_t* const input_end17 = input + ((input_words / 17) * 17);
	const uint64_t* const input_end = input + input_words;

	for (; input < input_end17; input += 17) {
		A[t] ^= input[t] & k2;

		ROUND(0x0000000000000001ULL); ROUND(0x0000000000008082ULL); ROUND(0x800000000000808AULL);
		ROUND(0x8000000080008000ULL); ROUND(0x000000000000808BULL); ROUND(0x0000000080000001ULL);
		ROUND(0x8000000080008081ULL); ROUND(0x8000000000008009ULL); ROUND(0x000000000000008AULL);
		ROUND(0x0000000000000088ULL); ROUND(0x0000000080008009ULL); ROUND(0x000000008000000AULL);
		ROUND(0x000000008000808BULL); ROUND(0x800000000000008BULL); ROUND(0x8000000000008089ULL);
		ROUND(0x8000000000008003ULL); ROUND(0x8000000000008002ULL); ROUND(0x8000000000000080ULL);
		ROUND(0x000000000000800AULL); ROUND(0x800000008000000AULL); ROUND(0x8000000080008081ULL);
		ROUND(0x8000000000008080ULL); ROUND(0x0000000080000001ULL); ROUND(0x8000000080008008ULL);
	}

	const uint32_t wordIndex = input_end - input;
	if (t < wordIndex) {
		A[t] ^= input[t];
	}

	if (t == 0) {
		uint64_t tail = 0;
		const uint8_t* p = (const uint8_t*)input_end;
		const uint32_t tail_size = input_size % sizeof(uint64_t);
		for (uint32_t i = 0; i < tail_size; ++i) {
			tail |= (uint64_t)(p[i]) << (i * 8);
		}

		A[wordIndex] ^= tail ^ ((uint64_t)(((uint64_t)(0x02 | (1 << 2))) << (tail_size * 8)));
		A[16] ^= 0x8000000000000000ULL;
	}

	sync();

	#pragma unroll(1)
	for (int i = 0; i < ROUNDS; ++i) {
		C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
		A[t] ^= C[s + 4] ^ R64(C[s + 1], 1, 63);
		C[t] = R64(A[at], ro0, ro1);
		A[t] = (C[t] ^ ((~C[c1]) & C[c2])) ^ (rc[i] & k1);
	}

	if (t < 4) {
		hashes[g * 4 + t] = A[t];
	}
}

__global__ void __launch_bounds__(32) sha3_initial(const uint8_t* input_data, uint32_t input_size, uint32_t nonce, uint64_t* hashes)
{
	const uint32_t t = threadIdx.x;
	const uint32_t g = blockIdx.x;

	if (t >= 25) {
		return;
	}

	const uint64_t* input = (const uint64_t*)(input_data);

	__shared__ uint64_t A[25];
	__shared__ uint64_t C[25];

	const uint32_t input_words = input_size / sizeof(uint64_t);
	A[t] = (t < input_words) ? input[t] : 0;

	if (t == 0) {
		((uint32_t*)A)[11] = nonce + g;

		const uint32_t tail_size = input_size % sizeof(uint64_t);
		A[input_words] ^= (uint64_t)(((uint64_t)(0x02 | (1 << 2))) << (tail_size * 8));
		A[16] ^= 0x8000000000000000ULL;
	}

	sync();

	const uint32_t s = t % 5;
	const int at = ppi[t][0];
	const int ro0 = ppi[t][1];
	const int ro1 = 64 - ro0;
	const int c1 = c[t][0];
	const int c2 = c[t][1];
	const uint64_t k1 = (t == 0) ? (uint64_t)(-1) : 0;

	ROUND(0x0000000000000001ULL); ROUND(0x0000000000008082ULL); ROUND(0x800000000000808AULL);
	ROUND(0x8000000080008000ULL); ROUND(0x000000000000808BULL); ROUND(0x0000000080000001ULL);
	ROUND(0x8000000080008081ULL); ROUND(0x8000000000008009ULL); ROUND(0x000000000000008AULL);
	ROUND(0x0000000000000088ULL); ROUND(0x0000000080008009ULL); ROUND(0x000000008000000AULL);
	ROUND(0x000000008000808BULL); ROUND(0x800000000000008BULL); ROUND(0x8000000000008089ULL);
	ROUND(0x8000000000008003ULL); ROUND(0x8000000000008002ULL); ROUND(0x8000000000000080ULL);
	ROUND(0x000000000000800AULL); ROUND(0x800000008000000AULL); ROUND(0x8000000080008081ULL);
	ROUND(0x8000000000008080ULL); ROUND(0x0000000080000001ULL); ROUND(0x8000000080008008ULL);

	if (t < 4) {
		hashes[g * 4 + t] = A[t];
	}
}

#undef ROUND

} // AstroBWT_Dero_HE
