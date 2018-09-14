#include "momentum_sgd_pt_op.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

__global__ void MomentumSGDPTKernel(
    const int N,
    const float* g,
    const float* m,
    float* ng,
    float* nm,
    const float* lr,
    const float momentum,
    const bool nesterov,
    float* param) {
  const float LR = lr[0];
  if (!nesterov) {
    CUDA_1D_KERNEL_LOOP(i, N) {
      const float adjusted_gradient =  g[i] + momentum * m[i];
      nm[i] = adjusted_gradient;
      ng[i] = adjusted_gradient;
      if (param) {
        param[i] -= LR * adjusted_gradient;
      }
    }
  } else {
    CUDA_1D_KERNEL_LOOP(i, N) {
      const float mi = m[i];
      const float mi_new = momentum * mi + g[i];
      nm[i] = mi_new;
      ng[i] = (1 + momentum) * mi_new - momentum * mi;
      if (param) {
        param[i] -= LR * ng[i];
      }
    }
  }
}

template <>
void momentum_sgd_pt_update<CUDAContext>(
    const int N,
    const float* g,
    const float* m,
    float* ng,
    float* nm,
    const float* lr,
    const float momentum,
    const bool nesterov,
    float* param,
    CUDAContext* context) {
  MomentumSGDPTKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      N, g, m, ng, nm, lr, momentum, nesterov, param);
}


template <typename SIndex>
__global__ void SparseMomentumSGDPTKernel(
    const size_t N,
    const size_t sz,
    const float momentum,
    const bool nesterov,
    float *param,
    float *param_mom,
    const SIndex *indices,
    const float *gradIn,
    float *gradOut,
    const float *lr)
{
  const float LR = lr[0];
  CUDA_1D_KERNEL_LOOP(i, N)
  {
    const size_t gradIdx = i;
    const SIndex index = indices[i / sz];
    const size_t paramIdx = index * sz + (i % sz);

    if (!nesterov)
    {
      const float adjusted_gradient = gradIn[gradIdx] +
          momentum * param_mom[paramIdx];
      gradOut[gradIdx] = adjusted_gradient;
      param_mom[paramIdx] = adjusted_gradient;
      param[paramIdx] -= LR * adjusted_gradient;
    } else {
      const float mom_old = param_mom[paramIdx];
      const float mom_new = gradIn[gradIdx] + momentum * mom_old;
      param_mom[paramIdx] = mom_new;
      const float adjusted_gradient = (1 + momentum) * mom_new -
          momentum * mom_old;
      gradOut[gradIdx] = adjusted_gradient;
      param[paramIdx] -= LR * adjusted_gradient;
    }
  }
}


// Specialization of DoRunWithType for CUDA
template <>
template <typename SIndex>
bool SparseMomentumSGDUpdatePTOp<float, CUDAContext>::DoRunWithType() {
  auto N = Input(GRAD).size();
  auto grad_slice_sz = Input(GRAD).size_from_dim(Input(INDICES).ndim());

  SparseMomentumSGDPTKernel<SIndex><<<
    CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
    context_.cuda_stream()>>>(
        N, grad_slice_sz,
        momentum_, nesterov_,
        Output(OUTPUT_PARAM)->template mutable_data<float>(),
        Output(OUTPUT_MOMENTUM)->template mutable_data<float>(),
        Input(INDICES).template data<SIndex>(),
        Input(GRAD).template data<float>(),
        Output(OUTPUT_GRAD)->template mutable_data<float>(),
        Input(LR).template data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(MomentumSGDPT, MomentumSGDPTOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(MomentumSGDUpdatePT, MomentumSGDUpdatePTOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SparseMomentumSGDUpdatePT, SparseMomentumSGDUpdatePTOp<float, CUDAContext>);

}
