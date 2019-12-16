/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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

#ifndef XMRIG_ALGORITHM_H
#define XMRIG_ALGORITHM_H


#include <cstddef>
#include <vector>


namespace xmrig {


class Algorithm
{
public:
    enum Id : int {
        INVALID = -1,
        CN_0,          // "cn/0"             CryptoNight (original).
        CN_1,          // "cn/1"             CryptoNight variant 1 also known as Monero7 and CryptoNightV7.
        CN_2,          // "cn/2"             CryptoNight variant 2.
        CN_R,          // "cn/r"             CryptoNightR (Monero's variant 4).
        CN_FAST,       // "cn/fast"          CryptoNight variant 1 with half iterations.
        CN_HALF,       // "cn/half"          CryptoNight variant 2 with half iterations (Masari/Torque).
        CN_XAO,        // "cn/xao"           CryptoNight variant 0 (modified, Alloy only).
        CN_RTO,        // "cn/rto"           CryptoNight variant 1 (modified, Arto only).
        CN_RWZ,        // "cn/rwz"           CryptoNight variant 2 with 3/4 iterations and reversed shuffle operation (Graft).
        CN_ZLS,        // "cn/zls"           CryptoNight variant 2 with 3/4 iterations (Zelerius).
        CN_DOUBLE,     // "cn/double"        CryptoNight variant 2 with double iterations (X-CASH).
        CN_GPU,        // "cn/gpu"           CryptoNight-GPU (Ryo).
        CN_LITE_0,     // "cn-lite/0"        CryptoNight-Lite variant 0.
        CN_LITE_1,     // "cn-lite/1"        CryptoNight-Lite variant 1.
        CN_HEAVY_0,    // "cn-heavy/0"       CryptoNight-Heavy (4 MB).
        CN_HEAVY_TUBE, // "cn-heavy/tube"    CryptoNight-Heavy (modified, TUBE only).
        CN_HEAVY_XHV,  // "cn-heavy/xhv"     CryptoNight-Heavy (modified, Haven Protocol only).
        CN_PICO_0,     // "cn-pico"          CryptoNight Turtle (TRTL)
        RX_0,          // "rx/0"             RandomX (reference configuration).
        RX_WOW,        // "rx/wow"           RandomWOW (Wownero).
        RX_LOKI,       // "rx/loki"          RandomXL (Loki).
        RX_ARQ,        // "rx/arq"           RandomARQ (Arqma).
        RX_SFX,        // "rx/sfx"           RandomSFX (Safex Cash).
        AR2_CHUKWA,    // "argon2/chukwa"    Argon2id (Chukwa).
        AR2_WRKZ,      // "argon2/wrkz"      Argon2id (WRKZ)
        MAX
    };

    enum Family : int {
        UNKNOWN,
        CN,
        CN_LITE,
        CN_HEAVY,
        CN_PICO,
        RANDOM_X,
        ARGON2
    };

    inline Algorithm() = default;
    inline Algorithm(Id id) : m_id(id)                {}
    inline Algorithm(int id) : m_id(id > INVALID && id < MAX ? static_cast<Id>(id) : INVALID) {}

    inline bool isCN() const                          { auto f = family(); return f == CN || f == CN_LITE || f == CN_HEAVY || f == CN_PICO; }
    inline bool isEqual(const Algorithm &other) const { return m_id == other.m_id; }
    inline bool isValid() const                       { return m_id != INVALID; }
    inline Family family() const                      { return family(m_id); }
    inline Id id() const                              { return m_id; }

    inline bool operator!=(Algorithm::Id id) const        { return m_id != id; }
    inline bool operator!=(const Algorithm &other) const  { return !isEqual(other); }
    inline bool operator==(Algorithm::Id id) const        { return m_id == id; }
    inline bool operator==(const Algorithm &other) const  { return isEqual(other); }
    inline operator Algorithm::Id() const                 { return m_id; }

    size_t l2() const
    {
        switch (m_id) {
        case RX_0:
        case RX_LOKI:
        case RX_SFX:
            return 0x40000;

        case RX_WOW:
            return 0x20000;

        case RX_ARQ:
            return 0x10000;

        default:
            break;
        }

        return 0;
    }

    size_t l3() const
    {
        constexpr size_t oneMiB = 0x100000;

        const Family f = family();

        if (isCN()) {
            switch (f) {
            case CN:
                return oneMiB * 2;

            case CN_LITE:
                return oneMiB;

            case CN_HEAVY:
                return oneMiB * 4;

            case CN_PICO:
                return oneMiB / 4;

            default:
                break;
            }

            return 0;
        }

        if (f == RANDOM_X) {
            switch (m_id) {
            case RX_0:
            case RX_LOKI:
            case RX_SFX:
                return oneMiB * 2;

            case RX_WOW:
                return oneMiB;

            case RX_ARQ:
                return oneMiB / 4;

            default:
                break;
            }
        }

        if (f == ARGON2) {
            switch (m_id) {
            case AR2_CHUKWA:
                return oneMiB / 2;

            case AR2_WRKZ:
                return oneMiB / 4;

            default:
                break;
            }
        }

        return 0;
    }

    static Family family(Id id)
    {
        switch (id) {
        case CN_0:
        case CN_1:
        case CN_2:
        case CN_R:
        case CN_FAST:
        case CN_HALF:
        case CN_XAO:
        case CN_RTO:
        case CN_RWZ:
        case CN_ZLS:
        case CN_DOUBLE:
        case CN_GPU:
            return CN;

        case CN_LITE_0:
        case CN_LITE_1:
            return CN_LITE;

        case CN_HEAVY_0:
        case CN_HEAVY_TUBE:
        case CN_HEAVY_XHV:
            return CN_HEAVY;

        case CN_PICO_0:
            return CN_PICO;

        case RX_0:
        case RX_WOW:
        case RX_LOKI:
        case RX_ARQ:
        case RX_SFX:
            return RANDOM_X;

        case AR2_CHUKWA:
        case AR2_WRKZ:
            return ARGON2;

        default:
            break;
        }

        return UNKNOWN;
    }

private:
    Id m_id = INVALID;
};


} /* namespace xmrig */


#endif /* XMRIG_ALGORITHM_H */
