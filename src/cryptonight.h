/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2019      Spudz76     <https://github.com/Spudz76>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "crypto/common/Algorithm.h"

#include <cstdint>
#include <cuda.h>


struct nvid_ctx {
    CUdevice cuDevice                   = 0;
    CUcontext cuContext                 = nullptr;
    CUmodule module                     = nullptr;
    CUfunction kernel                   = nullptr;
    xmrig::Algorithm algorithm          = xmrig::Algorithm::INVALID;
    uint64_t kernel_height              = 0;

    int device_id                       = 0;
    const char *device_name             = nullptr;
    int device_arch[2]                  { 0,};
    int device_mpcount                  = 0;
    int device_blocks                   = 0;
    int device_threads                  = 0;
    int device_bfactor                  = 0;
    int device_bsleep                   = 0;
    int device_clockRate                = 0;
    int device_memoryClockRate          = 0;
    size_t device_memoryTotal           = 0;
    size_t device_memoryFree            = 0;
    int device_pciBusID                 = 0;
    int device_pciDeviceID              = 0;
    int device_pciDomainID              = 0;
    uint32_t syncMode                   = 3;
    bool ready                          = false;

    uint32_t *d_input                   = nullptr;
    uint32_t inputlen                   = 0;
    uint32_t *d_result_count            = nullptr;
    uint32_t *d_result_nonce            = nullptr;
    uint32_t *d_long_state              = nullptr;
    uint64_t d_scratchpads_size         = 0;
    uint32_t *d_ctx_state               = nullptr;
    uint32_t *d_ctx_state2              = nullptr;
    uint32_t *d_ctx_a                   = nullptr;
    uint32_t *d_ctx_b                   = nullptr;
    uint32_t *d_ctx_key1                = nullptr;
    uint32_t *d_ctx_key2                = nullptr;
    uint32_t *d_ctx_text                = nullptr;

    uint32_t rx_batch_size              = 0;
    uint32_t *d_rx_dataset              = nullptr;
    uint32_t *d_rx_hashes               = nullptr;
    uint32_t *d_rx_entropy              = nullptr;
    uint32_t *d_rx_vm_states            = nullptr;
    uint32_t *d_rx_rounding             = nullptr;
};


int cuda_get_devicecount();
int cuda_get_runtime_version();
int cuda_get_driver_version();
int cuda_get_deviceinfo(nvid_ctx *ctx);
int cryptonight_gpu_init(nvid_ctx *ctx);
void cryptonight_extra_cpu_set_data(nvid_ctx *ctx, const void *data, size_t len);
void cryptonight_extra_cpu_prepare(nvid_ctx *ctx, uint32_t startNonce, const xmrig::Algorithm &algorithm);
void cryptonight_gpu_hash(nvid_ctx *ctx, const xmrig::Algorithm &algorithm, uint64_t height, uint32_t startNonce);
void cryptonight_extra_cpu_final(nvid_ctx *ctx, uint32_t startNonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, const xmrig::Algorithm &algorithm);

void randomx_prepare(nvid_ctx *ctx, const void *dataset, size_t dataset_size, uint32_t batch_size);

namespace RandomX_Arqma   { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }
namespace RandomX_Loki    { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }
namespace RandomX_Monero  { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }
namespace RandomX_Wownero { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }
