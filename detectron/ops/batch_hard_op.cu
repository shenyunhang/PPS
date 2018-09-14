#include "batch_hard_op.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"

namespace caffe2 {

namespace {}  // namespace

REGISTER_CUDA_OPERATOR(BatchHard, GPUFallbackOp);
REGISTER_CUDA_OPERATOR(BatchHardGradient, GPUFallbackOp);

}  // namespace caffe2
