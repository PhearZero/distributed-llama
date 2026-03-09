# How to Run on Rockchip NPU

Distributed Llama supports Rockchip NPU acceleration through the `rkllm` device.

## Prerequisites

1. A Rockchip device with an NPU (e.g., RK3588, RK3576).
2. Rockchip LLM SDK (`librkllmrt.so`) installed or available in the `src/rkllama/lib` directory.

### Library Setup (Mandatory)

Distributed Llama expects the Rockchip NPU libraries in `src/rkllama/lib/`. You must copy them from the manufacturer's SDKs included in this repository:

```bash
# 1. Create the library directory
mkdir -p src/rkllama/lib/

# 2. Copy the libraries (assuming Linux aarch64 for RK3588/RK3576)
cp rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so src/rkllama/lib/
cp rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so src/rkllama/lib/

# 3. Add to library path (or use run_npu.sh)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/src/rkllama/lib/
```

*Note: The updated `Makefile` and `examples/run_npu.sh` should handle this automatically using rpath.*

*Note: For Android or other architectures, use the corresponding folder in the SDK (e.g., `Android/arm64-v8a` or `Linux/armhf`).*

## Build

To build Distributed Llama with Rockchip NPU support, use the `DLLAMA_RKLLM` flag:

```bash
make DLLAMA_RKLLM=1
```

## Running

To use the NPU for inference, add the `--rkllm 1` argument to your command:

```bash
./dllama inference --model your_model.m --tokenizer tokenizer.t --rkllm 1 --prompt "Hello" --steps 128
```

## Current Limitations

The current integration is a baseline that enables the use of Rockchip's NPU libraries within the Distributed Llama ecosystem. 

* **Hybrid Execution**: If `--rkllm 1` is specified, the NPU is used as the primary compute device.
* **Model Format**: Distributed Llama normally uses its own `.m` format. To fully leverage the NPU, further integration with Rockchip's `.rkllm` format is required.
* **Fine-grained Offloading**: Distributed Llama now supports fine-grained offloading of specific LLM operations (currently Matrix Multiplication) to the Rockchip NPU using the RKNPU2 MatMul API. Other operations (like Softmax, RMS Norm, etc.) automatically fall back to the CPU, ensuring full model compatibility while providing NPU acceleration for bottleneck operations.

For more advanced Rockchip features, check the `rkllama` directory which contains specialized tools for Rockchip NPU.

Detailed examples for model conversion and usage can be found in [RKLLM_EXAMPLES.md](RKLLM_EXAMPLES.md).
