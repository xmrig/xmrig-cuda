/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
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


#include "cryptonight.h"
#include "version.h"
#include "xmrig-cuda.h"


#include <map>
#include <mutex>
#include <string>


static std::mutex mutex;
static std::map<int, std::string> errors;


static inline void saveError(int id, std::exception &ex)
{
    std::lock_guard<std::mutex> lock(mutex);
    errors[id] = ex.what();
}


static inline void resetError(int id)
{
    std::lock_guard<std::mutex> lock(mutex);
    errors.erase(id);
}


extern "C" {


bool cnHash(nvid_ctx *ctx, uint32_t startNonce, uint64_t height, uint64_t target, uint32_t *rescount, uint32_t *resnonce)
{
    resetError(ctx->device_id);

    try {
        cryptonight_extra_cpu_prepare(ctx, startNonce, ctx->algorithm);
        cryptonight_gpu_hash(ctx, ctx->algorithm, height, startNonce);
        cryptonight_extra_cpu_final(ctx, startNonce, target, rescount, resnonce, ctx->algorithm);
    }
    catch (std::exception &ex) {
        saveError(ctx->device_id, ex);

        return false;
    }

    return true;
}


bool deviceInit(nvid_ctx *ctx)
{
    resetError(ctx->device_id);

    if (ctx == nullptr) {
        return false;
    }

    int rc = 0;

    try {
        rc = cryptonight_gpu_init(ctx);
    }
    catch (std::exception &ex) {
        saveError(ctx->device_id, ex);

        return false;
    }

    return rc;
}


bool rxHash(nvid_ctx *ctx, uint32_t startNonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce)
{
    resetError(ctx->device_id);

    try {
        switch (ctx->algorithm.id()) {
        case xmrig::Algorithm::RX_0:
            RandomX_Monero::hash(ctx, startNonce, target, rescount, resnonce, ctx->rx_batch_size);
            break;

        case xmrig::Algorithm::RX_WOW:
            RandomX_Wownero::hash(ctx, startNonce, target, rescount, resnonce, ctx->rx_batch_size);
            break;

        case xmrig::Algorithm::RX_LOKI:
            RandomX_Loki::hash(ctx, startNonce, target, rescount, resnonce, ctx->rx_batch_size);
            break;

        case xmrig::Algorithm::RX_ARQ:
            RandomX_Arqma::hash(ctx, startNonce, target, rescount, resnonce, ctx->rx_batch_size);
            break;

        default:
            throw std::runtime_error("Unsupported algorithm");
            break;
        }
    }
    catch (std::exception &ex) {
        saveError(ctx->device_id, ex);

        return false;
    }

    return true;
}


bool rxPrepare(nvid_ctx *ctx, const void *dataset, size_t datasetSize, uint32_t batchSize)
{
    resetError(ctx->device_id);

    try {
        randomx_prepare(ctx, dataset, datasetSize, batchSize);
    }
    catch (std::exception &ex) {
        saveError(ctx->device_id, ex);

        return false;
    }

    return true;
}


bool setJob(nvid_ctx *ctx, const void *data, size_t size, int32_t algo)
{
    resetError(ctx->device_id);

    if (ctx == nullptr) {
        return false;
    }

    ctx->algorithm = algo;

    try {
        cryptonight_extra_cpu_set_data(ctx, data, size);
    }
    catch (std::exception &ex) {
        saveError(ctx->device_id, ex);

        return false;
    }

    return true;
}


const char *deviceName(nvid_ctx *ctx)
{
    return ctx->device_name;
}


const char *lastError(nvid_ctx *ctx)
{
    std::lock_guard<std::mutex> lock(mutex);

    return errors.count(ctx->device_id) ? errors[ctx->device_id].c_str() : nullptr;
}


const char *pluginVersion()
{
    return APP_VERSION;
}


int32_t deviceInfo(nvid_ctx *ctx, int32_t blocks, int32_t threads, int32_t algo)
{
    ctx->algorithm      = algo;
    ctx->device_blocks  = blocks;
    ctx->device_threads = threads;

    return cuda_get_deviceinfo(ctx);
}


int32_t deviceInt(nvid_ctx *ctx, DeviceProperty property)
{
    if (ctx == nullptr) {
        return 0;
    }

    switch (property) {
    case DeviceId:
        return ctx->device_id;

    case DeviceAlgorithm:
        return ctx->algorithm;

    case DeviceArchMajor:
        return ctx->device_arch[0];

    case DeviceArchMinor:
        return ctx->device_arch[1];

    case DeviceSmx:
        return ctx->device_mpcount;

    case DeviceBlocks:
        return ctx->device_blocks;

    case DeviceThreads:
        return ctx->device_threads;

    case DeviceBFactor:
        return ctx->device_bfactor;

    case DeviceBSleep:
        return ctx->device_bsleep;

    case DeviceClockRate:
        return ctx->device_clockRate;

    case DeviceMemoryClockRate:
        return ctx->device_memoryClockRate;

    case DevicePciBusID:
        return ctx->device_pciBusID;

    case DevicePciDeviceID:
        return ctx->device_pciDeviceID;

    case DevicePciDomainID:
        return ctx->device_pciDomainID;

    default:
        break;
    };

    return 0;
}


nvid_ctx *alloc(uint32_t id, int32_t bfactor, int32_t bsleep)
{
    auto ctx = new nvid_ctx();

    ctx->device_id      = static_cast<int>(id);
    ctx->device_bfactor = bfactor;
    ctx->device_bsleep  = bsleep;

    return ctx;
}


uint32_t deviceCount()
{
    return static_cast<uint32_t>(cuda_get_devicecount());
}


uint32_t deviceUint(nvid_ctx *ctx, DeviceProperty property)
{
    return static_cast<uint32_t>(deviceInt(ctx, property));
}


uint32_t version(Version version)
{
    switch (version) {
    case ApiVersion:
        return API_VERSION;

    case DriverVersion:
        return static_cast<uint32_t>(cuda_get_driver_version());

    case RuntimeVersion:
        return static_cast<uint32_t>(cuda_get_runtime_version());
    }

    return 0;
}


uint64_t deviceUlong(nvid_ctx *ctx, DeviceProperty property)
{
    if (ctx == nullptr) {
        return 0;
    }

    switch (property) {
    case DeviceMemoryTotal:
        return ctx->device_memoryTotal;

    case DeviceMemoryFree:
        return ctx->device_memoryFree;

    default:
        break;
    }

    return 0;
}


void init()
{
    cuInit(0);
}


void release(nvid_ctx *ctx)
{
    if (ctx == nullptr) {
        return;
    }

    if (ctx->ready) {
        cryptonight_extra_cpu_free(ctx);
    }

    delete [] ctx->device_name;

    delete ctx;
}


}
