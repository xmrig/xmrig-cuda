/* XMRig-CUDA — Animica SHA3-256 hashshare PoW
 *
 * Copyright 2026      ercmine     <https://github.com/ercmine>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   GPLv3 — see ../../LICENSE.
 */

#include "Animica/Animica.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>


namespace xmrig_cuda {
namespace Animica {


// ─── Keccak-f[1600] permutation (24 rounds) ─────────────────────────────
//
// Identical to the permutation in src/cuda_keccak.hpp; redeclared here
// in the Animica translation unit to keep this kernel independent of
// the CryptoNight build options that gate that file.

__constant__ uint64_t kKeccakRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__constant__ int kKeccakRho[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

__constant__ int kKeccakPi[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};


__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}


__device__ __forceinline__ void keccakf24(uint64_t s[25])
{
    uint64_t t, bc[5];
#pragma unroll 24
    for (int r = 0; r < 24; ++r) {
        // Theta
#pragma unroll 5
        for (int i = 0; i < 5; ++i) {
            bc[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];
        }
#pragma unroll 5
        for (int i = 0; i < 5; ++i) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) s[j + i] ^= t;
        }
        // Rho + Pi
        t = s[1];
#pragma unroll 24
        for (int i = 0; i < 24; ++i) {
            int j = kKeccakPi[i];
            bc[0] = s[j];
            s[j]  = rotl64(t, kKeccakRho[i]);
            t = bc[0];
        }
        // Chi
        for (int j = 0; j < 25; j += 5) {
#pragma unroll 5
            for (int i = 0; i < 5; ++i) bc[i] = s[j + i];
#pragma unroll 5
            for (int i = 0; i < 5; ++i) s[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        // Iota
        s[0] ^= kKeccakRC[r];
    }
}


// ─── Per-job device-side buffers ─────────────────────────────────────────

// Two read-only buffers and one writable result buffer. Allocated once
// per device context (lazily), bumped on prefix/target change, freed on
// release(ctx).

struct DeviceState {
    uint8_t  *prefix       = nullptr;   // 32 bytes
    uint8_t  *target_be    = nullptr;   // 32 bytes
    uint32_t *results      = nullptr;   // [count, nonce32, ...]
    size_t    resultsBytes = 0;
};

// One slot per (host-visible) CUDA device. xmrig-cuda allocates a
// nvid_ctx per device id, and a host process never has more than ~16
// CUDA devices; we cap at 16 for the static table. The mapping is
// keyed by ctx->device_id (a small int), so multiple ctxs targeting
// the same device share a slot — fine because xmrig-cuda is
// single-threaded per device during a hash() call.
static DeviceState g_state[16];


// ─── SHA3-256 search kernel ──────────────────────────────────────────────
//
// One thread per nonce. Hashes sha3_256(prefix || nonce_le8) and writes
// the (nonce, full digest) tuple into results[] when the BE 256-bit
// digest is ≤ target_be.

__global__ void animica_search_kernel(
    const uint8_t * __restrict__ prefix,
    const uint8_t * __restrict__ target_be,
    uint32_t startNonceLo,
    uint32_t startNonceHi,
    uint32_t *results)
{
    const uint64_t startNonce = ((uint64_t)startNonceHi << 32) | (uint64_t)startNonceLo;
    const uint64_t nonce      = startNonce + (uint64_t)((blockIdx.x * blockDim.x) + threadIdx.x);

    // Build the single 136-byte rate block: prefix(32) || nonce_le8 || pad || ... || pad|0x80.
    uint8_t buf[136];
#pragma unroll 17
    for (int i = 0; i < 17; ++i) ((uint64_t *)buf)[i] = 0;
#pragma unroll 32
    for (int i = 0; i < 32; ++i) buf[i] = prefix[i];
    // nonce LE
    buf[32] = (uint8_t)(nonce      );
    buf[33] = (uint8_t)(nonce >>  8);
    buf[34] = (uint8_t)(nonce >> 16);
    buf[35] = (uint8_t)(nonce >> 24);
    buf[36] = (uint8_t)(nonce >> 32);
    buf[37] = (uint8_t)(nonce >> 40);
    buf[38] = (uint8_t)(nonce >> 48);
    buf[39] = (uint8_t)(nonce >> 56);
    // SHA3 padding: 0x06 at end-of-message, 0x80 at last byte of rate.
    buf[40]  = 0x06;
    buf[135] |= 0x80;

    // Absorb + permute.
    uint64_t state[25];
#pragma unroll 25
    for (int i = 0; i < 25; ++i) state[i] = 0;
#pragma unroll 17
    for (int i = 0; i < 17; ++i) state[i] ^= ((uint64_t *)buf)[i];
    keccakf24(state);

    // Squeeze first 32 bytes.
    uint8_t digest[32];
#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        const uint64_t v = state[i];
        digest[i*8 + 0] = (uint8_t)(v      );
        digest[i*8 + 1] = (uint8_t)(v >>  8);
        digest[i*8 + 2] = (uint8_t)(v >> 16);
        digest[i*8 + 3] = (uint8_t)(v >> 24);
        digest[i*8 + 4] = (uint8_t)(v >> 32);
        digest[i*8 + 5] = (uint8_t)(v >> 40);
        digest[i*8 + 6] = (uint8_t)(v >> 48);
        digest[i*8 + 7] = (uint8_t)(v >> 56);
    }

    // BE compare against target.
    bool win = true;
    for (int i = 0; i < 32; ++i) {
        if (digest[i] < target_be[i]) { break; }
        if (digest[i] > target_be[i]) { win = false; break; }
    }
    if (!win) return;

    // Atomic-reserve a result slot. Layout per share: [nonce32].
    // We don't include the digest here — xmrig-cuda's hashOutput slot
    // is a uint32 nonce + the CudaWorker re-derives the digest on the
    // host side via the same sha3_256 path when it submits.
    const uint32_t idx = atomicAdd(&results[0], 1u);
    if (idx >= 8) return;   // hard cap — matches xmrig's per-round share cap
    results[1 + idx] = (uint32_t)(nonce & 0xFFFFFFFFu);
}


// ─── Host-side dispatcher ────────────────────────────────────────────────

static void cudaThrow(cudaError_t err, const char *what)
{
    if (err == cudaSuccess) return;
    std::string msg = "Animica CUDA ";
    msg += what;
    msg += ": ";
    msg += cudaGetErrorString(err);
    throw std::runtime_error(msg);
}


static void ensureAllocated(DeviceState &s)
{
    if (!s.prefix) {
        cudaThrow(cudaMalloc(&s.prefix,    32), "cudaMalloc(prefix)");
    }
    if (!s.target_be) {
        cudaThrow(cudaMalloc(&s.target_be, 32), "cudaMalloc(target)");
    }
    if (!s.results) {
        s.resultsBytes = (1 + 8) * sizeof(uint32_t);   // count + up to 8 nonces
        cudaThrow(cudaMalloc(&s.results,   s.resultsBytes), "cudaMalloc(results)");
    }
}


void hash(nvid_ctx *ctx, uint8_t *job_blob, uint64_t target,
          uint32_t startNonce, uint32_t *rescount, uint32_t *resnonce,
          uint32_t *skipped_hashes)
{
    cudaSetDevice(ctx->device_id);

    if (ctx->device_id < 0 || ctx->device_id >= 16) {
        throw std::runtime_error("Animica: device_id out of range (>=16)");
    }
    DeviceState &s = g_state[ctx->device_id];
    ensureAllocated(s);

    // Push prefix + target. job_blob[0..32) holds the SHA3 prefix
    // (Animica's prevhash); job_blob[32..40) is the nonce slot we
    // override per work-item, so we only ever copy the first 32 bytes.
    cudaThrow(cudaMemcpy(s.prefix, job_blob, 32, cudaMemcpyHostToDevice),
              "memcpy prefix");

    // Expand the host's high-64-bits BE target back into 32 BE bytes.
    uint8_t target_be_host[32] = {0};
    for (int i = 0; i < 8; ++i) {
        target_be_host[i] = (uint8_t)(target >> (8 * (7 - i)));
    }
    cudaThrow(cudaMemcpy(s.target_be, target_be_host, 32, cudaMemcpyHostToDevice),
              "memcpy target");

    // Reset result counter.
    cudaThrow(cudaMemset(s.results, 0, s.resultsBytes), "memset results");

    // Launch geometry: ctx->device_blocks * ctx->device_threads, same
    // shape KawPow uses. For a midrange RTX 3060 (28 SMs × 1024 threads
    // ≈ 28K) the kernel keeps the SMs saturated with one nonce per
    // thread; SHA3 is register-bound so occupancy is not the
    // bottleneck.
    const dim3 block(ctx->device_threads);
    const dim3 grid(ctx->device_blocks);

    const uint32_t startNonceLo = startNonce;
    const uint32_t startNonceHi = 0;

    animica_search_kernel<<<grid, block>>>(s.prefix, s.target_be,
                                             startNonceLo, startNonceHi, s.results);
    cudaThrow(cudaGetLastError(), "launch animica_search_kernel");
    cudaThrow(cudaDeviceSynchronize(), "sync animica_search_kernel");

    // Read back.
    uint32_t host_results[1 + 8] = {0};
    cudaThrow(cudaMemcpy(host_results, s.results, sizeof(host_results),
                          cudaMemcpyDeviceToHost), "memcpy results");

    *rescount = host_results[0] > 8 ? 8 : host_results[0];
    if (*rescount > 0) {
        // xmrig's submit path reads one 4-byte nonce from resnonce —
        // surface the first share; subsequent shares in this round are
        // dropped by xmrig anyway.
        *resnonce = host_results[1];
    }
    if (skipped_hashes) *skipped_hashes = 0;
}


}  // namespace Animica
}  // namespace xmrig_cuda
