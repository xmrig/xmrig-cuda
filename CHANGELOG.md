# v6.4.0
- [#70](https://github.com/xmrig/xmrig-cuda/pull/70) RandomX: removed `rx/loki` algorithm.
- Added CMake option `-DWITH_DRIVER_API=OFF` to disable CUDA Driver API and NVRTC, required for `cn/r` and `kawpow` algorithms.

# v6.3.2
- [#65](https://github.com/xmrig/xmrig-cuda/pull/65) Fixed broken AstroBWT.

# v6.3.1
- [#62](https://github.com/xmrig/xmrig-cuda/pull/62) Fixed broken RandomX (regression since v6.2.1).

# v6.3.0
- [#59](https://github.com/xmrig/xmrig-cuda/pull/59) Added support for upcoming Haven offshore fork.
- Fixed build with recent CUDA 11.

# v6.2.1
- [#54](https://github.com/xmrig/xmrig-cuda/pull/54) Optimized KawPow, about 2% hashrate improvement, 10% faster DAG initialization.
- [#55](https://github.com/xmrig/xmrig-cuda/pull/55) Added fast job switching for KawPow, almost zero stale shares.

# v6.2.0
- [#52](https://github.com/xmrig/xmrig-cuda/pull/52) Added new algorithm `cn/ccx` for Conceal.
- [#53](https://github.com/xmrig/xmrig-cuda/pull/53) Fixed build with CUDA 11.

# v6.1.0
- [#48](https://github.com/xmrig/xmrig-cuda/pull/48) Optimized AstroBWT, approximately 3 times faster.
- [#51](https://github.com/xmrig/xmrig-cuda/pull/51) Reduced memory usage for KawPow.

# v6.0.0
- [#1694](https://github.com/xmrig/xmrig/pull/1694) Added support for KawPow algorithm (Ravencoin) on AMD/NVIDIA.

# v3.0.0
- **ABI changed, minimum supported XMRig version now is 5.11.0.**
- [#41](https://github.com/xmrig/xmrig-cuda/pull/41) Added AstroBWT algorithm support.

# v2.2.0
- [#1578](https://github.com/xmrig/xmrig/pull/1578) Added new `rx/keva` algorithm for upcoming Kevacoin fork.

# v2.1.0
- [#1466](https://github.com/xmrig/xmrig/pull/1466) Added `cn-pico/tlo` algorithm.
- Added alternative relaxed API (algorithm passed as string).

# v2.0.2
- [#27](https://github.com/xmrig/xmrig-cuda/pull/27) Added RandomSFX (`rx/sfx`) algorithm for Safex Cash.
- [#28](https://github.com/xmrig/xmrig-cuda/pull/28) Added RandomV (`rx/v`) algorithm for *new* MoneroV.

# v2.0.1-beta
- [#10](https://github.com/xmrig/xmrig-cuda/pull/10) Fixed compatibility with CUDA 8, RandomX support not tested and potentially broken with this CUDA version.
- [#1276](https://github.com/xmrig/xmrig/issues/1276) Fixed maximum threads count.

# v2.0.0-beta
- **ABI changed, minimum supported XMRig version now is 4.6.0.**
- [#5](https://github.com/xmrig/xmrig-cuda/pull/5) Optimized RandomX.
- [#6](https://github.com/xmrig/xmrig-cuda/issues/6) Fixed compatibility with some old systems.
- [#7](https://github.com/xmrig/xmrig-cuda/pull/7) Added support for option `dataset_host` for 2 GB GPUs.
- [#8](https://github.com/xmrig/xmrig-cuda/pull/8) RandomX: fixed random kernel launch errors with some configurations.

# v1.0.0-beta
- Initial version.
