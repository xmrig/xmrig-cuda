/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "crypto/common/Algorithm.h"
#include "crypto/cn/CnAlgo.h"


#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>


#ifdef _MSC_VER
#   define strcasecmp  _stricmp
#endif


namespace xmrig_cuda {


struct AlgoName
{
    const char *name;
    const Algorithm::Id id;
};


static AlgoName const algorithm_names[] = {
#   ifdef XMRIG_ALGO_RANDOMX
    { "rx/0",             Algorithm::RX_0            },
    { "rx/wow",           Algorithm::RX_WOW          },
    { "rx/arq",           Algorithm::RX_ARQ          },
    { "rx/sfx",           Algorithm::RX_SFX          },
    { "rx/keva",          Algorithm::RX_KEVA         },
#   endif
    { "cn/0",             Algorithm::CN_0            },
    { "cn/1",             Algorithm::CN_1            },
    { "cn/2",             Algorithm::CN_2            },
    { "cn/fast",          Algorithm::CN_FAST         },
    { "cn/half",          Algorithm::CN_HALF         },
    { "cn/xao",           Algorithm::CN_XAO          },
    { "cn/rto",           Algorithm::CN_RTO          },
    { "cn/rwz",           Algorithm::CN_RWZ          },
    { "cn/zls",           Algorithm::CN_ZLS          },
    { "cn/double",        Algorithm::CN_DOUBLE       },
    { "cn/ccx",           Algorithm::CN_CCX          },
#   ifdef XMRIG_ALGO_CN_LITE
    { "cn-lite/0",        Algorithm::CN_LITE_0       },
    { "cn-lite/1",        Algorithm::CN_LITE_1       },
#   endif
#   ifdef XMRIG_ALGO_CN_HEAVY
    { "cn-heavy/0",       Algorithm::CN_HEAVY_0      },
    { "cn-heavy/xhv",     Algorithm::CN_HEAVY_XHV    },
    { "cn-heavy/tube",    Algorithm::CN_HEAVY_TUBE   },
#   endif
#   ifdef XMRIG_ALGO_CN_PICO
    { "cn-pico",          Algorithm::CN_PICO_0       },
    { "cn-pico/tlo",      Algorithm::CN_PICO_TLO     },
#   endif
#   ifdef XMRIG_ALGO_CN_FEMTO
    { "cn/upx2",          Algorithm::CN_UPX2          },
    // Algo names from other miners
    { "cn-extremelite/upx2", Algorithm::CN_UPX2       },
    { "cryptonight-upx/2",   Algorithm::CN_UPX2       },
#   endif
#   ifdef XMRIG_ALGO_ASTROBWT
    { "astrobwt",         Algorithm::ASTROBWT_DERO   },
#   endif
#   ifdef XMRIG_ALGO_KAWPOW
    { "kawpow",           Algorithm::KAWPOW_RVN      },
#   endif
#   ifdef XMRIG_ALGO_CN_R
    { "cn/r",             Algorithm::CN_R            },
#   endif
};


} /* namespace xmrig_cuda */



xmrig_cuda::Algorithm::Id xmrig_cuda::Algorithm::parseName(const char *name)
{
    if (name == nullptr || strlen(name) < 1) {
        return INVALID;
    }

    for (const auto &item : algorithm_names) {
        if ((strcasecmp(name, item.name) == 0)) {
            return item.id;
        }
    }

    return INVALID;
}
