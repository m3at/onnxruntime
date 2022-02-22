// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <unordered_map>
#include <string>

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000

using namespace onnxruntime::cuda;

class QuantizeWithOrder final : public CudaKernel {
 public:
  QuantizeWithOrder(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_input_;
  int order_output_;
};

class DequantizeWithOrder final : public CudaKernel {
 public:
  DequantizeWithOrder(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_input_;
  int order_output_;
  ONNX_NAMESPACE::TensorProto_DataType to_;
};

class QOrderedMatMul final : public CudaKernel {
 public:
  QOrderedMatMul(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_A_;
  int order_B_;
  int order_Y_;
};

cublasLtOrder_t GetCublasLtOrderAttr(const OpKernelInfo& info, const char* order_attr);

int64_t CalcLeadingDimensionLt(int64_t rows, int64_t cols, cublasLtOrder_t order);

class CublasLtMMAlgoMap {
 public:
  static CublasLtMMAlgoMap& instance();

  void GetAlgo(cublasLtMatmulAlgo_t& algo, const cudaDeviceProp& device_prop,
               int batch_count, int m, int n, int k,
               cublasLtOrder_t weight_order, cublasLtOrder_t input_output_order = CUBLASLT_ORDER_COL32) const;

  CublasLtMMAlgoMap(const CublasLtMMAlgoMap&) = delete;

  CublasLtMMAlgoMap& operator=(const CublasLtMMAlgoMap&) = delete;

 private:
  CublasLtMMAlgoMap();

  ~CublasLtMMAlgoMap() {}

 private:
  struct CublasLtMatmulAlgoInfo {
    int algoId, customOption, tile, splitK_val, swizzle, reductionScheme, workspaceSize, stages;
    float exec_time;
  };

  std::unordered_map<std::string, CublasLtMatmulAlgoInfo> best_algos_;
};

void QOrdered_MatMul(
    cublasLtHandle_t cublasLt_handle, cudaStream_t stream, const cudaDeviceProp& device_prop,
    int batchCount, int m, int n, int k,
    const float* alpha,
    const int8_t* A, int64_t batch_stride_A,
    const int8_t* B, int64_t batch_stride_B,
    const float* beta,
    const int8_t* C, int64_t ldc,
    int8_t* D, int64_t batch_stride_D,
    cublasLtOrder_t weight_order);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
