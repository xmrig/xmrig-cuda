# xmrig-cuda

[![Github All Releases](https://img.shields.io/github/downloads/xmrig/xmrig-cuda/total.svg)](../../releases)
[![GitHub release](https://img.shields.io/github/release/xmrig/xmrig-cuda/all.svg)](../../releases/latest)
[![GitHub Release Date](https://img.shields.io/github/release-date/xmrig/xmrig-cuda.svg)](../../releases/latest)
[![GitHub license](https://img.shields.io/github/license/xmrig/xmrig-cuda.svg)](./LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/xmrig/xmrig-cuda.svg)](../../stargazers)
[![GitHub forks](https://img.shields.io/github/forks/xmrig/xmrig-cuda.svg)](../../network)

This repository is NVIDIA CUDA plugin for [XMRig](../../../xmrig) miner v4.5+ and it adds support for NVIDIA GPUs in [XMRig](../../../xmrig) miner.

Main reasons why this plugin is separated project is:
1. CUDA support is optional, not all users need it, but it is very heavy.
2. CUDA has very strict compiler version requirements, it may conflicts with CPU mining code, for example now possible build the miner with gcc on Windows (CUDA works only with MSVC).


## Table of Contents

 - [Windows usage](#windows-usage)
 - [Advanced](#advanced)
 - [Linux usage](#linux-usage)

## Windows usage

* [Download](../../releases) plugin, you must choose CUDA version, usually it recent version (CUDA 10.1), but builds with older CUDA version also provided, alternative you can build the plugin from source.
* Place **`xmrig-cuda.dll`** and other dll files near to **`xmrig.exe`**.
* Edit **`config.json`** to enable CUDA support.
```
{
   ...
   "cuda": {
      "enabled": true,
      ...
   }
   ...
}
```
### Advanced
Path to plugin can be specified via `loader` option:
```
{
   ...
   "cuda": {
      "enabled": true,
      "loader": "c:/some/path/xmrig-cuda.dll",
      ...
   }
   ...
}
```
Due of restrictions of JSON format directory separator must be written in Linux style `/` or escaped `\\`.

## Linux usage
Linux usage almost same with Windows expept we don't provide binaries and you must build the plugin form source and name of plugin is different **`libxmrig-cuda.so`**.
