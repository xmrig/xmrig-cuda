# xmrig-cuda
This repository contains the CUDA plugin for the XMRig miner, which provides support for NVIDIA GPUs.

This plugin is a separate project because of the main reasons listed below:
1. Not all users require CUDA support, and it is an optional feature.
2. CUDA has strict compiler version requirements that may conflict with the CPU mining code. For example, CUDA doesn't support GCC on Windows.


## Windows

* To [download](https://github.com/xmrig/xmrig-cuda/releases) the plugin, you must choose the appropriate CUDA version. Generally, the latest version (12.4) is all you need, unless you have very old GPUs. Windows builds are available for every major CUDA release. Alternatively, you can [build](https://xmrig.com/docs/miner/build/windows) the plugin from the source.
* Place **`xmrig-cuda.dll`** and other dll files near to **`xmrig.exe`**.
* Edit **`config.json`** enable the plugin.
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
You can specify the path to the plugin using the `loader` option.
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
Due to JSON format restrictions, the directory separator must be written in Linux style `/` or escaped `\\`.

## Linux
Linux usage is almost the same as Windows except we don't provide binaries and you must build the plugin from the source and the name of the plugin is different **`libxmrig-cuda.so`**.

## macOS
CUDA no longer supports macOS, which means that the plugin also does not support it.
