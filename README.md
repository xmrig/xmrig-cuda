# xmrig-cuda
This repository is NVIDIA CUDA plugin for XMRig miner v4.5+ and it adds support for NVIDIA GPUs in XMRig miner.

Main reasons why this plugin is separated project is:
1. CUDA support is optional, not all users need it, but it is very heavy.
2. CUDA has very strict compiler version requirements, it may conflicts with CPU mining code, for example now possible build the miner with gcc on Windows (CUDA works only with MSVC).


## Windows usage

* [Download](https://github.com/xmrig/xmrig-cuda/releases) plugin, you must choose CUDA version, usually it recent version (CUDA 10.1), but builds with older CUDA version also provided, alternative you can build the plugin from source.
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
