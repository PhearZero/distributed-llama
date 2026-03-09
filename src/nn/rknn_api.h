#ifndef RKNN_API_H
#define RKNN_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RKNN_MAX_DIMS 16
#define RKNN_MAX_NAME_LEN 256

#ifdef __arm__
typedef uint32_t rknn_context;
#else
typedef uint64_t rknn_context;
#endif

typedef rknn_context rknn_matmul_ctx;

typedef enum _rknn_tensor_type {
    RKNN_TENSOR_FLOAT32 = 0,
    RKNN_TENSOR_FLOAT16,
    RKNN_TENSOR_INT8,
    RKNN_TENSOR_UINT8,
    RKNN_TENSOR_INT16,
    RKNN_TENSOR_UINT16,
    RKNN_TENSOR_INT32,
    RKNN_TENSOR_UINT32,
    RKNN_TENSOR_INT64,
    RKNN_TENSOR_BOOL,
    RKNN_TENSOR_INT4,
    RKNN_TENSOR_BFLOAT16,
    RKNN_TENSOR_TYPE_MAX
} rknn_tensor_type;

typedef enum _rknn_tensor_format {
    RKNN_TENSOR_NCHW = 0,
    RKNN_TENSOR_NHWC,
    RKNN_TENSOR_NC1HWC2,
    RKNN_TENSOR_UNDEFINED,
    RKNN_TENSOR_FORMAT_MAX
} rknn_tensor_format;

typedef enum _rknn_matmul_type {
    RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 = 1,
    RKNN_INT8_MM_INT8_TO_INT32         = 2,
    RKNN_INT8_MM_INT8_TO_INT8          = 3,
    RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16 = 4,
    RKNN_FLOAT16_MM_INT8_TO_FLOAT32    = 5,
    RKNN_FLOAT16_MM_INT8_TO_FLOAT16    = 6,
    RKNN_FLOAT16_MM_INT4_TO_FLOAT32    = 7,
    RKNN_FLOAT16_MM_INT4_TO_FLOAT16    = 8,
    RKNN_INT8_MM_INT8_TO_FLOAT32       = 9,
    RKNN_INT4_MM_INT4_TO_INT16         = 10,
    RKNN_INT8_MM_INT4_TO_INT32         = 11,
    RKNN_FLOAT16_MM_INT4_TO_BFLOAT16   = 12,
    RKNN_INT8_MM_INT4_TO_FLOAT16       = 15,
} rknn_matmul_type;

typedef enum _rknn_core_mask {
    RKNN_NPU_CORE_AUTO = 0,
    RKNN_NPU_CORE_0 = 1,
    RKNN_NPU_CORE_1 = 2,
    RKNN_NPU_CORE_2 = 4,
    RKNN_NPU_CORE_0_1 = RKNN_NPU_CORE_0 | RKNN_NPU_CORE_1,
    RKNN_NPU_CORE_0_1_2 = RKNN_NPU_CORE_0_1 | RKNN_NPU_CORE_2,
    RKNN_NPU_CORE_ALL = 0xffff,
    RKNN_NPU_CORE_UNDEFINED,
} rknn_core_mask;

typedef struct _rknn_tensor_attr {
    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[RKNN_MAX_DIMS];
    char name[RKNN_MAX_NAME_LEN];
    uint32_t n_elems;
    uint32_t size;
    rknn_tensor_format fmt;
    rknn_tensor_type type;
    int8_t qnt_type;
    int8_t fl;
    int32_t zp;
    float scale;
    uint32_t w_stride;
    uint32_t size_with_stride;
    uint8_t pass_through;
    uint32_t h_stride;
} rknn_tensor_attr;

typedef struct _rknn_matmul_tensor_attr {
    char name[RKNN_MAX_NAME_LEN];
    uint32_t n_dims;
    uint32_t dims[RKNN_MAX_DIMS];
    uint32_t size;
    rknn_tensor_type type;
} rknn_matmul_tensor_attr;

typedef struct _rknn_matmul_io_attr {
    rknn_matmul_tensor_attr A;
    rknn_matmul_tensor_attr B;
    rknn_matmul_tensor_attr C;
} rknn_matmul_io_attr;

typedef struct _rknn_matmul_shape {
    int32_t M;
    int32_t K;
    int32_t N;
} rknn_matmul_shape;

typedef struct _rknn_matmul_info {
    int32_t M;
    int32_t K;
    int32_t N;
    rknn_matmul_type type;
    int16_t B_layout;
    int16_t B_quant_type;
    int16_t AC_layout;
    int16_t AC_quant_type;
    int32_t iommu_domain_id;
    int16_t group_size;
    int8_t reserved[34];
} rknn_matmul_info;

typedef struct _rknn_tensor_mem {
    void* virt_addr;
    uint64_t phys_addr;
    int32_t fd;
    int32_t offset;
    uint32_t size;
    uint32_t flags;
    void* priv_data;
} rknn_tensor_mem;

int rknn_matmul_create(rknn_matmul_ctx* ctx, rknn_matmul_info* info, rknn_matmul_io_attr* io_attr);
int rknn_matmul_set_io_mem(rknn_matmul_ctx ctx, rknn_tensor_mem* mem, rknn_matmul_tensor_attr* attr);
int rknn_matmul_run(rknn_matmul_ctx ctx);
int rknn_matmul_destroy(rknn_matmul_ctx ctx);
int rknn_matmul_set_core_mask(rknn_matmul_ctx context, rknn_core_mask core_mask);
int rknn_matmul_set_dynamic_shape(rknn_matmul_ctx ctx, rknn_matmul_shape* shape);
int rknn_B_normal_layout_to_native_layout(void* B_input, void* B_output, int K, int N, rknn_matmul_info* info);

rknn_tensor_mem* rknn_create_mem(rknn_context ctx, uint32_t size);
int rknn_destroy_mem(rknn_context ctx, rknn_tensor_mem* mem);

#ifdef __cplusplus
}
#endif

#endif
