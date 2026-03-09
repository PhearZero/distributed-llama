#ifndef NN_RKLLM_HPP
#define NN_RKLLM_HPP

#include "nn-executor.hpp"
#include "nn-cpu-ops.hpp"
#include "rknn_api.h"
#include <map>

class NnRkllmDevice : public NnDevice {
private:
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
    NnNetExecution *netExecution;
public:
    NnRkllmDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution);
    ~NnRkllmDevice() override;
    NnUint maxNThreads() override;
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;

    // Support for CPU fallback with SRC_BUFFER
    std::unique_ptr<NnDevice> cpuFallbackDevice;
};

class NnRkllmDeviceSegment : public NnDeviceSegment {
private:
    NnNetConfig *netConfig;
    NnUint segmentIndex;
    NnSegmentConfig *segmentConfig;
    NnNetExecution *netExecution;

    struct MatMulContext {
        rknn_matmul_ctx ctx;
        rknn_matmul_io_attr io_attr;
        rknn_tensor_mem* mem_A;
        rknn_tensor_mem* mem_B;
        rknn_tensor_mem* mem_C;
        bool initialized = false;
        NnUint currentBatchSize = 0;
        NnByte *nativeWeight = nullptr;
    };

    std::map<NnUint, MatMulContext> matmulContexts;
    std::vector<NnCpuOpForward> cpuOpForward;
    std::vector<NnCpuOpContext> cpuOpContexts;

public:
    NnByte **cpuBuffers = nullptr;
    NnBufferConfig *cpuBufferConfigs = nullptr;
    NnByte *cpuBufferFlags = nullptr;

    NnRkllmDeviceSegment(NnNetConfig *netConfig, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution);
    ~NnRkllmDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
};

#endif
