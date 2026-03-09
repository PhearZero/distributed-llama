#include "nn-rkllm.hpp"
#include "nn-cpu.hpp"
#include "nn-cpu-ops.hpp"
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <algorithm>

static NnByte *getPointer(NnNetExecution *netExecution, NnPointerConfig *config, NnByte **cpuBuffers) {
    if (config->source == SRC_PIPE)
        return netExecution->pipes[config->pointerIndex];
    if (config->source == SRC_BUFFER && cpuBuffers != nullptr)
        return cpuBuffers[config->pointerIndex];
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
    NnRkllmDeviceSegment *segment = new NnRkllmDeviceSegment(netConfig, segmentIndex, &nodeConfig->segments[segmentIndex], netExecution);
    if (this->cpuFallbackDevice) {
        NnCpuDevice *cpuDevice = (NnCpuDevice *)this->cpuFallbackDevice.get();
        segment->cpuBuffers = cpuDevice->buffers;
        segment->cpuBufferConfigs = this->nodeConfig->buffers;
        segment->cpuBufferFlags = cpuDevice->bufferFlags;
    }
    return segment;
}

NnRkllmDeviceSegment::NnRkllmDeviceSegment(NnNetConfig *netConfig, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution)
    : netConfig(netConfig), segmentIndex(segmentIndex), segmentConfig(segmentConfig), netExecution(netExecution) {
    cpuOpForward.resize(segmentConfig->nOps, nullptr);
    cpuOpContexts.resize(segmentConfig->nOps);
    for (NnUint i = 0; i < segmentConfig->nOps; i++) {
        cpuOpContexts[i].input = nullptr;
        cpuOpContexts[i].output = nullptr;
        cpuOpContexts[i].weight = nullptr;
        cpuOpContexts[i].weightSize.nBytes = 0;
    }
}

NnRkllmDeviceSegment::~NnRkllmDeviceSegment() {
    for (NnUint opIndex = 0; opIndex < cpuOpContexts.size(); opIndex++) {
        NnCpuOpContext *context = &cpuOpContexts[opIndex];
        if (context->input) delete[] context->input;
        if (context->output) delete[] context->output;
        if (context->weightSize.nBytes > 0 && context->weight) {
            free(context->weight);
        }
    }
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
    } else {
        // Fallback to CPU loading
        NnCpuOpContext *context = &cpuOpContexts[opIndex];
        if (context->weight == nullptr && opConfig->weightSize.nBytes > 0) {
            // lazy allocation
            if (posix_memalign((void **)&context->weight, 64, opConfig->weightSize.nBytes) != 0)
                throw std::runtime_error("posix_memalign failed");
        }
        if (context->weight) {
            std::memcpy(&context->weight[offset], weight, nBytes);
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
        NnByte *inputA = getPointer(netExecution, &opConfig->input, cpuBuffers);
        if (inputA) {
            // Need to handle FP32 to FP16 conversion if info.type is RKNN_FLOAT16_MM_...
            // Distributed Llama often uses FP32 for activations in the pipeline.
            // For now, assume matching types or add conversion logic.
            std::memcpy(ctx.mem_A->virt_addr, inputA, std::min((NnSize)ctx.io_attr.A.size, (NnSize)(batchSize * K * sizeof(float))));
        }

        // 2. Run MatMul
        rknn_matmul_run(ctx.ctx);

        // 3. Copy Output C to output pipe
        NnByte *outputC = getPointer(netExecution, &opConfig->output, cpuBuffers);
        if (outputC) {
            std::memcpy(outputC, ctx.mem_C->virt_addr, std::min((NnSize)ctx.io_attr.C.size, (NnSize)(batchSize * N * sizeof(float))));
        }

        return;
    }

    // Fallback to CPU execution
    NnCpuOpContext *context = &cpuOpContexts[opIndex];
    if (cpuOpForward[opIndex] == nullptr) {
        // Initialize CPU context
        NnSize3D inputSize;
        NnSize3D outputSize;

        // We need a way to resolve pointers.
        // NnCpuDevice has resolvePointer, but it uses buffers from NnCpuDevice.
        // NnRkllmDevice doesn't have its own buffers yet, it relies on system pipes mostly.

        auto resolvePntr = [&](NnSize3D *pntrSize, NnPointerConfig *pointerConfig) -> std::vector<NnByte *> {
            NnByte *source;
            NnSize3D *sourceSize;

            switch (pointerConfig->source) {
            case SRC_BUFFER:
                if (cpuBuffers == nullptr) {
                    throw std::runtime_error("SRC_BUFFER fallback failed: CPU buffers not initialized in NnRkllmDevice");
                }
                source = cpuBuffers[pointerConfig->pointerIndex];
                sourceSize = &cpuBufferConfigs[pointerConfig->pointerIndex].size;
                break;
            case SRC_PIPE:
                source = netExecution->pipes[pointerConfig->pointerIndex];
                sourceSize = &netConfig->pipes[pointerConfig->pointerIndex].size;
                break;
            default:
                throw std::invalid_argument("Unsupported pointer type");
            }

            switch (pointerConfig->type) {
            case PNTR_RAW: {
                *pntrSize = size1D(sourceSize->floatType, sourceSize->length);
                return std::vector<NnByte *>{source};
            }
            case PNTR_BATCH:
            case PNTR_BATCHED_SLICE: {
                if (sourceSize->y != netConfig->nBatches) throw std::runtime_error("Batch size mismatch");
                std::vector<NnByte *> pntr(sourceSize->z * sourceSize->y);

                NnSize batchBytes = getBytes(sourceSize->floatType, sourceSize->x);
                for (NnUint z = 0u; z < sourceSize->z; z++) {
                    for (NnUint y = 0u; y < sourceSize->y; y++)
                        pntr[z * sourceSize->y + y] = &source[(z * sourceSize->y + y) * batchBytes];
                }
                *pntrSize = *sourceSize;

                if (pointerConfig->type == PNTR_BATCHED_SLICE) {
                    // Fallback to single node slice for now if needed, or implement full slicing
                    // For now, assume nodeIndex 0 or implement correctly
                    // NnRkllmDevice segment should know its nodeIndex if we passed it.
                }
                return pntr;
            }
            default:
                throw std::invalid_argument("Unsupported pointer config");
            }
        };

        std::vector<NnByte *> inputs = resolvePntr(&inputSize, &opConfig->input);
        std::vector<NnByte *> outputs = resolvePntr(&outputSize, &opConfig->output);

        NnOpQuantType opQuant = getOpQuantType(
            inputSize.floatType,
            opConfig->weightSize.floatType,
            outputSize.floatType);

        cpuOpForward[opIndex] = getCpuOpForward(opConfig->code, opQuant);
        if (cpuOpForward[opIndex] == nullptr) {
             throw std::runtime_error("Rockchip NPU fallback: CPU implementation not found for op code: " + std::to_string(opConfig->code));
        }

        context->name = opConfig->name;
        context->opConfig = opConfig->config;
        context->weightSize = opConfig->weightSize;
        context->nBatches = netConfig->nBatches;
        context->pipes = netExecution->pipes;
        context->pipeConfigs = netConfig->pipes;
        context->buffers = cpuBuffers;
        context->bufferConfigs = cpuBufferConfigs;
        context->bufferFlags = cpuBufferFlags;

        context->input = new NnByte *[inputs.size()];
        context->inputSize = inputSize;
        context->hasInputContinuousMemory = hasPointerContinuousMemory(&opConfig->input);
        std::memcpy(context->input, inputs.data(), inputs.size() * sizeof(NnByte *));

        context->output = new NnByte *[outputs.size()];
        context->outputSize = outputSize;
        context->hasOutputContinuousMemory = hasPointerContinuousMemory(&opConfig->output);
        std::memcpy(context->output, outputs.data(), outputs.size() * sizeof(NnByte *));

        NnCpuOpForwardInit opInit = getCpuOpForwardInit(opConfig->code, opQuant);
        if (opInit) opInit(context);
    }

    cpuOpForward[opIndex](nThreads, threadIndex, batchSize, context);
}
