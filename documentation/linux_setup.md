# Linux Setup Guide

This will go over the Linux setup for running XMRig on CUDA. 

# Dependencies 

Before beginning, ensure you have CUDA installed. You can check your version with:
```bash
nvcc --version
```
which will output the CUDA version if installed. You can also run 
```bash
nvidia-smi
``` 
which should show a description of your GPU if everything is setup correctly. 

If the above commands indicate you don't have CUDA installed, then follow the instructions from [the official Nvidia installation website](https://developer.nvidia.com/cuda-downloads?target_os=Linux).

Next, ensure you have the necessary dependencies installed:
```bash
sudo apt update
sudo apt install build-essential cmake libssl-dev libhwloc-dev libcurl4-openssl-dev
sudo apt install nvidia-cuda-toolkit
```

# Setup

Begin by cloning both repos into adjacent folders in the same directory.

```bash
git clone https://github.com/xmrig/xmrig-cuda.git
git clone https://github.com/xmrig/xmrig.git
```

You're folder structure should be organized as follows:
```
xmrig/
xmrig-cuda/
```

Next, build the CUDA plugin code from source to create the plugin `.so` file we need. We build in a `build` directory we create in the repo as follows:

```bash
cd xmrig-cuda
mkdir build
cd build
cmake ..
make
```

Next, we build the main XMRig project from source in a similiar manner:
```bash
cd ../../xmrig
mkdir build
cd build
cmake ..
make
```

Now copy the plugin into this directory with the executable:
```bash
cp ../../xmrig-cuda/build/libxmrig-cuda.so .
```

Finally create your config JSON following [the XMRig official config wizard](https://xmrig.com/wizard) and copy the downloaded `config.json` file (or create the file and paste the contents) in this build directory.

To summarize, your executable file, `xmrig`, should exist in the `xmrig/build` directory and this same folder should contain the `config.json` and `.so` CUDA plugin:
```
xmrig/
    | ...
    | build/
           | config.json
           | libxmrig-cuda.so
           | xmrig
           | ...
xmrig-cuda/
```
You can now delete the `xmrig-cuda` directory if you want. 

The miner can be ran with `./xmrig` from the `xmrig/build` folder.

# All in One Copy + Paste

The following combines all commands into a single copy + paste

```bash
git clone https://github.com/xmrig/xmrig-cuda.git
git clone https://github.com/xmrig/xmrig.git
cd xmrig-cuda
mkdir build
cd build
cmake ..
make
cd ../../xmrig
mkdir build
cd build
cmake ..
make
cp ../../xmrig-cuda/build/libxmrig-cuda.so .
```