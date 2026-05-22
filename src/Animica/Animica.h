/* XMRig-CUDA
 * Copyright 2026      ercmine     <https://github.com/ercmine>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 */

#ifndef XMRIG_CUDA_ANIMICA_H
#define XMRIG_CUDA_ANIMICA_H


#include <cstdint>

#include "cryptonight.h"     // brings in nvid_ctx


namespace xmrig_cuda {
namespace Animica {


/**
 * One round of Animica hashshare PoW on the GPU.
 *
 * job_blob layout (40 bytes, prepared host-side by xmrig):
 *   [0..32)  prefix   — the SHA3-256 prefix (Animica's stratum-v1
 *                       carries this as the 32-byte prevhash field).
 *   [32..40) nonce    — 8-byte LE nonce SLOT; not read here — each
 *                       work-item derives its own nonce from
 *                       startNonce + global_id(0).
 *
 * target is the high 64 bits of the 256-bit big-endian target.
 * The kernel expands it to 32 bytes BE on-device and compares the
 * full SHA3-256 digest byte-by-byte.
 *
 * On success up to one share is written to *resnonce and *rescount is
 * set to 1. If no work-item beats the target in this batch, rescount
 * stays 0. *skipped_hashes is set to the number of hashes the kernel
 * had to throw away because the launch geometry overshot startNonce's
 * remaining range (matches the KawPow contract).
 */
void hash(nvid_ctx *ctx, uint8_t *job_blob, uint64_t target,
          uint32_t startNonce, uint32_t *rescount, uint32_t *resnonce,
          uint32_t *skipped_hashes);


}  // namespace Animica
}  // namespace xmrig_cuda


#endif // XMRIG_CUDA_ANIMICA_H
