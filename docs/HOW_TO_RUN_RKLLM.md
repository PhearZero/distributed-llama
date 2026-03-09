# How to Run on Rockchip NPU

Distributed Llama supports Rockchip NPU acceleration through the `rkllm` device.

## Prerequisites

1. A Rockchip device with an NPU (e.g., RK3588, RK3576).
2. Rockchip LLM SDK (`librkllmrt.so`) installed or available in the `src/rkllama/lib` directory.

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
* **Fine-grained Offloading**: Distributed Llama now supports fine-grained offloading of specific LLM operations (currently Matrix Multiplication) to the Rockchip NPU using the RKNPU2 MatMul API. This allows for hybrid execution where the NPU accelerates the most compute-intensive parts while the CPU handles other operations.

For more advanced Rockchip features, check the `rkllama` directory which contains specialized tools for Rockchip NPU.

Detailed examples for model conversion and usage can be found in [RKLLM_EXAMPLES.md](RKLLM_EXAMPLES.md).
