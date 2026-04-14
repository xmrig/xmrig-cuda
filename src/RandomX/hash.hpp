#pragma once

/*
Copyright (c) 2019 SChernykh

This file is part of RandomX CUDA.

RandomX CUDA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX CUDA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX CUDA.  If not, see<http://www.gnu.org/licenses/>.
*/

__global__ void find_shares(const void* hashes, uint64_t target, uint32_t* shares)
{
    const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t* p = (const uint64_t*)hashes;

    if (p[global_index * 4 + 3] < target) {
        const uint32_t idx = atomicInc(shares, 0xFFFFFFFF) + 1;
        if (idx < 10) {
            shares[idx] = global_index;
        }
    }
}

#if RANDOMX_TWEAK_V2_COMMITMENT
__global__ void blake2b_hash_commitment_single(void *hashes, const void *blockTemplate, uint32_t blockTemplate_len, uint32_t start_nonce, uint32_t nonce_offset)
{
    const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t *input = (const uint8_t *) blockTemplate;
    const uint8_t *raw_hash = ((const uint8_t *) hashes) + global_index * 32;

    uint64_t block[16] = { 0 };
    uint8_t *msg = (uint8_t *) block;

    for (uint32_t i = 0; i < blockTemplate_len; ++i) {
        msg[i] = input[i];
    }

    const uint32_t nonce = start_nonce + global_index;
    msg[nonce_offset + 0] = static_cast<uint8_t>(nonce);
    msg[nonce_offset + 1] = static_cast<uint8_t>(nonce >> 8);
    msg[nonce_offset + 2] = static_cast<uint8_t>(nonce >> 16);
    msg[nonce_offset + 3] = static_cast<uint8_t>(nonce >> 24);

    for (uint32_t i = 0; i < RANDOMX_HASH_SIZE; ++i) {
        msg[blockTemplate_len + i] = raw_hash[i];
    }

    const uint32_t total_len = blockTemplate_len + RANDOMX_HASH_SIZE;
    if (total_len % sizeof(uint64_t)) {
        block[total_len / sizeof(uint64_t)] &= uint64_t(-1) >> (64 - (total_len % sizeof(uint64_t)) * 8);
    }

    uint64_t commitment[4] = { 0 };
    blake2b_process_single_block<RANDOMX_HASH_SIZE>(commitment, block, total_len);

    uint64_t *out = ((uint64_t *) hashes) + global_index * (RANDOMX_HASH_SIZE / sizeof(uint64_t));
    out[0] = commitment[0];
    out[1] = commitment[1];
    out[2] = commitment[2];
    out[3] = commitment[3];
}

__global__ void blake2b_hash_commitment_big(void *hashes, const void *blockTemplate, uint32_t blockTemplate_len, uint32_t start_nonce, uint32_t nonce_offset)
{
    const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t *input = (const uint8_t *) blockTemplate;
    const uint8_t *raw_hash = ((const uint8_t *) hashes) + global_index * 32;

    uint64_t message[64] = { 0 };
    uint8_t *msg = (uint8_t *) message;

    for (uint32_t i = 0; i < blockTemplate_len; ++i) {
        msg[i] = input[i];
    }

    for (uint32_t i = 0; i < RANDOMX_HASH_SIZE; ++i) {
        msg[blockTemplate_len + i] = raw_hash[i];
    }

    const uint32_t total_len = blockTemplate_len + RANDOMX_HASH_SIZE;
    uint64_t commitment[4] = { 0 };
    blake2b_512_process_big_block<RANDOMX_HASH_SIZE>(commitment, message, total_len, start_nonce + global_index, nonce_offset);

    uint64_t *out = ((uint64_t *) hashes) + global_index * (RANDOMX_HASH_SIZE / sizeof(uint64_t));
    out[0] = commitment[0];
    out[1] = commitment[1];
    out[2] = commitment[2];
    out[3] = commitment[3];
}

__global__ void blake2b_hash_commitment_double(void *hashes, const void *blockTemplate, uint32_t blockTemplate_len, uint32_t start_nonce, uint32_t nonce_offset)
{
    const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t *input = (const uint8_t *) blockTemplate;
    const uint8_t *raw_hash = ((const uint8_t *) hashes) + global_index * 32;

    uint64_t blocks[32] = { 0 };
    uint8_t *msg = (uint8_t *) blocks;

    for (uint32_t i = 0; i < blockTemplate_len; ++i) {
        msg[i] = input[i];
    }

    const uint32_t nonce = start_nonce + global_index;
    msg[nonce_offset + 0] = static_cast<uint8_t>(nonce);
    msg[nonce_offset + 1] = static_cast<uint8_t>(nonce >> 8);
    msg[nonce_offset + 2] = static_cast<uint8_t>(nonce >> 16);
    msg[nonce_offset + 3] = static_cast<uint8_t>(nonce >> 24);

    for (uint32_t i = 0; i < RANDOMX_HASH_SIZE; ++i) {
        msg[blockTemplate_len + i] = raw_hash[i];
    }

    const uint32_t total_len = blockTemplate_len + RANDOMX_HASH_SIZE;
    if (total_len % sizeof(uint64_t)) {
        blocks[total_len / sizeof(uint64_t)] &= uint64_t(-1) >> (64 - (total_len % sizeof(uint64_t)) * 8);
    }

    uint64_t commitment[4] = { 0 };
    blake2b_512_process_double_block<RANDOMX_HASH_SIZE>(commitment, blocks, blocks, total_len);

    uint64_t *out = ((uint64_t *) hashes) + global_index * (RANDOMX_HASH_SIZE / sizeof(uint64_t));
    out[0] = commitment[0];
    out[1] = commitment[1];
    out[2] = commitment[2];
    out[3] = commitment[3];
}
#endif

void hash(nvid_ctx *ctx, uint32_t nonce, uint32_t nonce_offset, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size)
{
    if (ctx->inputlen <= 128) {
        CUDA_CHECK_KERNEL(ctx->device_id, blake2b_initial_hash << <batch_size / 32, 32 >> > (ctx->d_rx_hashes, ctx->d_input, ctx->inputlen, nonce));
    }
    else if (ctx->inputlen <= 256) {
        CUDA_CHECK_KERNEL(ctx->device_id, blake2b_initial_hash_double << <batch_size / 32, 32 >> > (ctx->d_rx_hashes, ctx->d_input, ctx->inputlen, nonce));
    }
    else {
        CUDA_CHECK_KERNEL(ctx->device_id, blake2b_initial_hash_big << <batch_size / 32, 32 >> > (ctx->d_rx_hashes, ctx->d_input, ctx->inputlen, nonce, nonce_offset));
    }

    CUDA_CHECK_KERNEL(ctx->device_id, fillAes1Rx4<RANDOMX_SCRATCHPAD_L3, false, 64><<<batch_size / 32, 32 * 4>>>(ctx->d_rx_hashes, ctx->d_long_state, batch_size));
    CUDA_CHECK(ctx->device_id, cudaMemset(ctx->d_rx_rounding, 0, batch_size * sizeof(uint32_t)));

    for (size_t i = 0; i < RANDOMX_PROGRAM_COUNT; ++i) {
        CUDA_CHECK_KERNEL(ctx->device_id, fillAes4Rx4<ENTROPY_SIZE, false><<<batch_size / 32, 32 * 4>>>(ctx->d_rx_hashes, ctx->d_rx_entropy, batch_size));

        CUDA_CHECK_KERNEL(ctx->device_id, init_vm<8><<<batch_size / 4, 4 * 8>>>(ctx->d_rx_entropy, ctx->d_rx_vm_states));
        for (int j = 0, n = 1 << ctx->device_bfactor; j < n; ++j) {
            CUDA_CHECK_KERNEL(ctx->device_id, execute_vm<8, false><<<batch_size / 2, 2 * 8>>>(ctx->d_rx_vm_states, ctx->d_rx_rounding, ctx->d_long_state, ctx->d_rx_dataset, batch_size, RANDOMX_PROGRAM_ITERATIONS >> ctx->device_bfactor, j == 0, j == n - 1));
        }

        if (i == RANDOMX_PROGRAM_COUNT - 1) {
            CUDA_CHECK_KERNEL(ctx->device_id, hashAes1Rx4<RANDOMX_SCRATCHPAD_L3, 192, VM_STATE_SIZE, 64><<<batch_size / 32, 32 * 4>>>(ctx->d_long_state, ctx->d_rx_vm_states, batch_size));
            CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_registers<REGISTERS_SIZE, VM_STATE_SIZE, 32><<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_rx_vm_states));
        } else {
            CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_registers<REGISTERS_SIZE, VM_STATE_SIZE, 64><<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_rx_vm_states));
        }
    }

#if RANDOMX_TWEAK_V2_COMMITMENT
    const uint32_t commitment_size = ctx->inputlen + RANDOMX_HASH_SIZE;
    if (commitment_size <= 128) {
        CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_commitment_single<<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_input, ctx->inputlen, nonce, nonce_offset));
    }
    else if (commitment_size <= 256) {
        CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_commitment_double<<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_input, ctx->inputlen, nonce, nonce_offset));
    }
    else {
        CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_commitment_big<<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_input, ctx->inputlen, nonce, nonce_offset));
    }
#endif

    CUDA_CHECK(ctx->device_id, cudaMemset(ctx->d_result_nonce, 0, 10 * sizeof(uint32_t)));
    CUDA_CHECK_KERNEL(ctx->device_id, find_shares<<<batch_size / 32, 32>>>(ctx->d_rx_hashes, target, ctx->d_result_nonce));
    CUDA_CHECK(ctx->device_id, cudaDeviceSynchronize());

    CUDA_CHECK(ctx->device_id, cudaMemcpy(resnonce, ctx->d_result_nonce, 10 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    *rescount = resnonce[0];
    if (*rescount > 9) {
        *rescount = 9;
    }

    for (uint32_t i = 0; i < *rescount; i++) {
        resnonce[i] = resnonce[i + 1] + nonce;
    }
}
