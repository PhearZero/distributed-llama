#include "nn-rkllm.hpp"
#include "nn-cpu-ops.hpp"
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <algorithm>

static NnByte *getPointer(NnNetExecution *netExecution, NnPointerConfig *config) {
    if (config->source == SRC_PIPE)
        return netExecution->pipes[config->pointerIndex];
    // SRC_BUFFER is not easily accessible here as we don't have nodeConfig buffers memory.
    // In distributed-llama, buffers are usually managed by the device or global memory.
    // For now, we'll focus on SRC_PIPE which is used for activations.
    return nullptr;
}

NnRkllmDevice::NnRkllmDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution)
    : netConfig(netConfig), nodeConfig(nodeConfig), netExecution(netExecution) {
    printf("NnRkllmDevice created (Fine-grained Rockchip NPU acceleration enabled)\n");
}

NnRkllmDevice::~NnRkllmDevice() {}

NnUint NnRkllmDevice::maxNThreads() {
    return 1;
}

NnDeviceSegment *NnRkllmDevice::createSegment(NnUint segmentIndex) {
    return new NnRkllmDeviceSegment(netConfig, segmentIndex, &nodeConfig->segments[segmentIndex], netExecution);
}

NnRkllmDeviceSegment::NnRkllmDeviceSegment(NnNetConfig *netConfig, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution)
    : netConfig(netConfig), segmentIndex(segmentIndex), segmentConfig(segmentConfig), netExecution(netExecution) {
}

NnRkllmDeviceSegment::~NnRkllmDeviceSegment() {
    for (auto& pair : matmulContexts) {
        MatMulContext& ctx = pair.second;
        if (ctx.initialized) {
            rknn_destroy_mem(ctx.ctx, ctx.mem_A);
            rknn_destroy_mem(ctx.ctx, ctx.mem_B);
            rknn_destroy_mem(ctx.ctx, ctx.mem_C);
            rknn_matmul_destroy(ctx.ctx);
        }
        if (ctx.nativeWeight) {
            delete[] ctx.nativeWeight;
        }
    }
}

void NnRkllmDeviceSegment::loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
    if (opConfig->code == OP_MATMUL) {
        if (matmulContexts.find(opIndex) == matmulContexts.end()) {
            MatMulContext ctx;
            ctx.initialized = false;
            ctx.nativeWeight = nullptr;
            matmulContexts[opIndex] = ctx;
        }

        MatMulContext &ctx = matmulContexts[opIndex];
        const NnUint K = opConfig->weightSize.y;
        const NnUint N = opConfig->weightSize.x;

        // Rockchip NPU MatMul requires special layout for weights (native layout).
        // Since we don't have the context yet (depends on M), we can't easily use rknn_B_normal_layout_to_native_layout
        // until we know the SoC.
        // However, Distributed Llama weights are typically loaded once.
        // For now, let's keep the original weight and we'll transform it on first forward if needed,
        // or just copy it if we already initialized.

        if (ctx.initialized) {
            // If already initialized, we can directly copy/transform to mem_B
            // But usually loadWeight happens before forward.
            if (opConfig->weightSize.floatType == F_16) {
                rknn_matmul_info info;
                std::memset(&info, 0, sizeof(info));
                info.K = K;
                info.N = N;
                info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
                rknn_B_normal_layout_to_native_layout(weight, ctx.mem_B->virt_addr, K, N, &info);
            } else {
                std::memcpy(ctx.mem_B->virt_addr, weight, std::min((NnSize)ctx.io_attr.B.size, nBytes));
            }
        } else {
            // Buffer the weight if it's the full weight
            if (offset == 0 && nBytes == opConfig->weightSize.nBytes) {
                if (ctx.nativeWeight) delete[] ctx.nativeWeight;
                ctx.nativeWeight = new NnByte[nBytes];
                std::memcpy(ctx.nativeWeight, weight, nBytes);
            }
        }
    }
}

void NnRkllmDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) {
    NnOpConfig *opConfig = &segmentConfig->ops[opIndex];

    if (opConfig->code == OP_MATMUL) {
        MatMulContext &ctx = matmulContexts[opIndex];
        const NnUint K = opConfig->weightSize.y;
        const NnUint N = opConfig->weightSize.x;

        if (!ctx.initialized) {
            rknn_matmul_info info;
            std::memset(&info, 0, sizeof(info));
            info.M = batchSize;
            info.K = K;
            info.N = N;
            info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32; // Default to FP16 -> FP32 for now
            info.B_layout = 1; // Native layout for B
            info.AC_layout = 0; // Normal layout for A and C

            int ret = rknn_matmul_create(&ctx.ctx, &info, &ctx.io_attr);
            if (ret != 0) {
                throw std::runtime_error("rknn_matmul_create failed: " + std::to_string(ret));
            }

            ctx.mem_A = rknn_create_mem(ctx.ctx, ctx.io_attr.A.size);
            ctx.mem_B = rknn_create_mem(ctx.ctx, ctx.io_attr.B.size);
            ctx.mem_C = rknn_create_mem(ctx.ctx, ctx.io_attr.C.size);

            rknn_matmul_set_io_mem(ctx.ctx, ctx.mem_A, &ctx.io_attr.A);
            rknn_matmul_set_io_mem(ctx.ctx, ctx.mem_B, &ctx.io_attr.B);
            rknn_matmul_set_io_mem(ctx.ctx, ctx.mem_C, &ctx.io_attr.C);

            // Set core mask for RK3588 (Core ALL = 0xffff)
            rknn_matmul_set_core_mask(ctx.ctx, RKNN_NPU_CORE_ALL);

            // Convert and copy weight to mem_B
            if (ctx.nativeWeight) {
                if (opConfig->weightSize.floatType == F_16) {
                    rknn_B_normal_layout_to_native_layout(ctx.nativeWeight, ctx.mem_B->virt_addr, K, N, &info);
                } else {
                    std::memcpy(ctx.mem_B->virt_addr, ctx.nativeWeight, std::min((NnSize)ctx.io_attr.B.size, (NnSize)opConfig->weightSize.nBytes));
                }
                delete[] ctx.nativeWeight;
                ctx.nativeWeight = nullptr;
            }

            ctx.initialized = true;
            ctx.currentBatchSize = batchSize;
        } else if (ctx.currentBatchSize != batchSize) {
            // Update dynamic shape if batch size changed
            rknn_matmul_shape shape;
            shape.M = batchSize;
            shape.K = K;
            shape.N = N;
            int ret = rknn_matmul_set_dynamic_shape(ctx.ctx, &shape);
            if (ret != 0) {
                throw std::runtime_error("rknn_matmul_set_dynamic_shape failed: " + std::to_string(ret));
            }
            ctx.currentBatchSize = batchSize;
        }

        // 1. Copy Input A (Activations) to mem_A
        NnByte *inputA = getPointer(netExecution, &opConfig->input);
        if (inputA) {
            // Need to handle FP32 to FP16 conversion if info.type is RKNN_FLOAT16_MM_...
            // Distributed Llama often uses FP32 for activations in the pipeline.
            // For now, assume matching types or add conversion logic.
            std::memcpy(ctx.mem_A->virt_addr, inputA, std::min((NnSize)ctx.io_attr.A.size, (NnSize)(batchSize * K * sizeof(float))));
        }

        // 2. Run MatMul
        rknn_matmul_run(ctx.ctx);

        // 3. Copy Output C to output pipe
        NnByte *outputC = getPointer(netExecution, &opConfig->output);
        if (outputC) {
            std::memcpy(outputC, ctx.mem_C->virt_addr, std::min((NnSize)ctx.io_attr.C.size, (NnSize)(batchSize * N * sizeof(float))));
        }

        return;
    }

    throw std::runtime_error("Rockchip NPU: Fine-grained offloading only implemented for MatMul. Op code: " + std::to_string(opConfig->code));
}
