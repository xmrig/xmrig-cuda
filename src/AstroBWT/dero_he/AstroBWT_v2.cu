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


#include "cryptonight.h"
#include "cuda_device.hpp"


#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

static __device__ __forceinline__ void sync()
{
#   if (__CUDACC_VER_MAJOR__ >= 9)
    __syncwarp();
#   else
    __syncthreads();
#   endif
}

#include "sha3.h"
#include "salsa20.h"
#include "BWT.h"
namespace tmp {
#include "../dero/sha3.h"
}


#if (__CUDACC_VER_MAJOR__ >= 11)
#include <cub/device/device_segmented_radix_sort.cuh>
#else
#include "3rdparty/cub/device/device_segmented_radix_sort.cuh"
#endif


static constexpr uint32_t DATA_STRIDE = 10240;


void astrobwt_prepare_v2(nvid_ctx *ctx, uint32_t batch_size)
{
    if (batch_size == ctx->astrobwt_intensity)
        return;

    ctx->astrobwt_intensity = batch_size;

    const uint32_t BATCH_SIZE = batch_size;
    const uint32_t ALLOCATION_SIZE = BATCH_SIZE * DATA_STRIDE;

    CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_shares));
    CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_salsa20_keys));
    CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_bwt_data));
    CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_indices));
    CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_tmp_indices));
    CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_offsets_begin));
    CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_offsets_end));

    CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_shares, 16 * sizeof(uint32_t)));
    CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_salsa20_keys, BATCH_SIZE * 32));
    CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_bwt_data, ALLOCATION_SIZE));
    CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_indices, ALLOCATION_SIZE * 8));
    CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_tmp_indices, ALLOCATION_SIZE * sizeof(uint32_t) + 65536));
    CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_offsets_begin, BATCH_SIZE * sizeof(uint32_t)));
    CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_offsets_end, BATCH_SIZE * sizeof(uint32_t)));

    std::vector<uint32_t> v(BATCH_SIZE);

    for (uint32_t i = 0; i < BATCH_SIZE; ++i) {
        v[i] = i * DATA_STRIDE;
    }
    cudaMemcpy(ctx->astrobwt_offsets_begin, v.data(), BATCH_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);

    for (uint32_t i = 0; i < BATCH_SIZE; ++i) {
        v[i] = i * DATA_STRIDE + 9973;
    }
    cudaMemcpy(ctx->astrobwt_offsets_end, v.data(), BATCH_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);

    CUDA_CHECK(ctx->device_id, cudaDeviceSynchronize());
}

namespace AstroBWT_Dero_HE {

void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce)
{
    const uint32_t BATCH_SIZE = ctx->astrobwt_intensity;
    const uint32_t ALLOCATION_SIZE = BATCH_SIZE * DATA_STRIDE;

    const uint32_t zero = 0;
    CUDA_CHECK(ctx->device_id, cudaMemcpy((uint8_t*)(ctx->astrobwt_shares), &zero, sizeof(zero), cudaMemcpyHostToDevice));

    CUDA_CHECK_KERNEL(ctx->device_id, sha3_initial<<<BATCH_SIZE, 32>>>((uint8_t*)ctx->d_input, ctx->inputlen, nonce, (uint64_t*)ctx->astrobwt_salsa20_keys));
    CUDA_CHECK_KERNEL(ctx->device_id, Salsa20_XORKeyStream<<<BATCH_SIZE, 32>>>((uint32_t*)ctx->astrobwt_salsa20_keys, (uint32_t*)ctx->astrobwt_bwt_data));

    uint16_t* p = (uint16_t*)ctx->astrobwt_indices;
    uint16_t* keys_in = p;
    uint16_t* keys_out = p + ALLOCATION_SIZE;
    uint16_t* values_in = p + ALLOCATION_SIZE * 2;
    uint16_t* values_out = p + ALLOCATION_SIZE * 3;

    CUDA_CHECK_KERNEL(ctx->device_id, BWT_preprocess<<<BATCH_SIZE, 1024>>>(
        (uint8_t*)ctx->astrobwt_bwt_data,
        keys_in,
        values_in
    ));

    size_t temp_storage_bytes = ALLOCATION_SIZE * 4 + 65536;
    cub::DeviceSegmentedRadixSort::SortPairs(
        ctx->astrobwt_tmp_indices,
        temp_storage_bytes,
        keys_in,
        keys_out,
        values_in,
        values_out,
        BATCH_SIZE * DATA_STRIDE,
        BATCH_SIZE,
        (uint32_t*)ctx->astrobwt_offsets_begin,
        (uint32_t*)ctx->astrobwt_offsets_end
    );

    CUDA_CHECK_KERNEL(ctx->device_id, BWT_fix_order<<<BATCH_SIZE, 1024>>>(
        (uint8_t*)ctx->astrobwt_bwt_data,
        keys_out,
        values_out
    ));

    CUDA_CHECK_KERNEL(ctx->device_id, sha3<<<BATCH_SIZE, 32>>>((uint8_t*)values_out, (uint64_t*)ctx->astrobwt_salsa20_keys));
    CUDA_CHECK_KERNEL(ctx->device_id, find_shares<<<BATCH_SIZE / 32, 32>>>((uint64_t*)ctx->astrobwt_salsa20_keys, target, (uint32_t*)ctx->astrobwt_shares));

    CUDA_CHECK(ctx->device_id, cudaDeviceSynchronize());

    uint32_t shares[11];
    CUDA_CHECK(ctx->device_id, cudaMemcpy(shares, ctx->astrobwt_shares, sizeof(shares), cudaMemcpyDeviceToHost));

    if (shares[0] > 10)
        shares[0] = 10;

    *rescount = shares[0];
    for (uint32_t i = 0; i < shares[0]; ++i) {
        resnonce[i] = nonce + shares[i + 1];
    }

    ctx->astrobwt_processed_hashes = BATCH_SIZE;
}

} // AstroBWT_Dero_HE
