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

__global__ __launch_bounds__(1024)
void BWT_preprocess(const uint8_t* datas, uint16_t* keys_in, uint16_t* values_in)
{
	const uint32_t data_offset = blockIdx.x * 10240;

	datas += data_offset;
	keys_in += data_offset;
	values_in += data_offset;

	for (uint32_t i = threadIdx.x; i < 9973; i += 1024)
	{
		keys_in[i] = (static_cast<uint16_t>(datas[i]) << 8) | datas[i + 1];
		values_in[i] = static_cast<uint16_t>(i);
	}
}

__device__ __forceinline__ void fix_order(const uint8_t* input, uint32_t a, uint32_t b, uint16_t* values_in)
{
	const uint32_t value_in_a = values_in[a];
	const uint32_t value_in_b = values_in[b];

	const uint32_t value_a =
		(static_cast<uint32_t>(input[value_in_a + 2]) << 24) |
		(static_cast<uint32_t>(input[value_in_a + 3]) << 16) |
		(static_cast<uint32_t>(input[value_in_a + 4]) << 8) |
		static_cast<uint32_t>(input[value_in_a + 5]);

	const uint32_t value_b =
		(static_cast<uint32_t>(input[value_in_b + 2]) << 24) |
		(static_cast<uint32_t>(input[value_in_b + 3]) << 16) |
		(static_cast<uint32_t>(input[value_in_b + 4]) << 8) |
		static_cast<uint32_t>(input[value_in_b + 5]);

	if (value_a > value_b)
	{
		values_in[a] = value_in_b;
		values_in[b] = value_in_a;
	}
}

__global__ __launch_bounds__(1024)
void BWT_fix_order(const uint8_t* datas, uint16_t* keys_in, uint16_t* values_in)
{
	const uint32_t tid = threadIdx.x;
	const uint32_t gid = blockIdx.x;

	const uint32_t data_offset = gid * 10240;
	const uint8_t* input = datas + data_offset;

	const uint32_t N = 9973;

	keys_in += data_offset;
	values_in += data_offset;

	for (uint32_t i = tid, N1 = N - 1; i < N1; i += 1024)
	{
		const uint16_t value = keys_in[i];
		if (value == (keys_in[i + 1]))
		{
			if (i && (value == keys_in[i - 1]))
				continue;

			uint32_t n = i + 2;
			while ((n < N) && (value == keys_in[n]))
				++n;

			for (uint32_t j = i; j < n; ++j)
				for (uint32_t k = j + 1; k < n; ++k)
					fix_order(input, j, k, values_in);
		}
	}
}

__global__ void __launch_bounds__(32) find_shares(const uint64_t* hashes, uint64_t target, uint32_t* shares)
{
	const uint32_t global_index = blockIdx.x * 32 + threadIdx.x;

	if (hashes[global_index * 4 + 3] < target)
	{
		const int idx = atomicAdd((int*)(shares), 1) + 1;
		if (idx < 16)
			shares[idx] = global_index;
	}
}

} // AstroBWT_Dero_HE
