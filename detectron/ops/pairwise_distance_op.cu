#include "caffe2/core/context_gpu.h"
#include "pairwise_distance_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void PairWiseDistanceKernel(const int N, const int D, const T* X,
                                       T* Z) {
  CUDA_1D_KERNEL_LOOP(i, N * N) {
    int p = i / N;
    int q = i % N;
    T dist = 0;
    for (int d = 0; d < D; d++) {
      T sub = X[p * D + d] - X[q * D + d];
      dist += sub * sub;
    }
    // Z[m * N + n] = sqrt(dist);
    Z[p * N + q] = dist;
  }
}
}  // namespace

template <>
bool PairWiseDistanceOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);

  CAFFE_ENFORCE_EQ(X.dim(), 2);

  int N = X.dim32(0);
  int D = X.dim32(1);

  auto* Z = Output(0);
  Z->Resize(N, N);

  PairWiseDistanceKernel << <CAFFE_GET_BLOCKS(N * N), CAFFE_CUDA_NUM_THREADS, 0,
                             context_.cuda_stream()>>>
      (N, D, X.data<float>(), Z->mutable_data<float>());
  return true;
}

namespace {

template <typename T>
__global__ void PairWiseDistanceGradientKernel(const int N, const int D,
                                               const T* X, const T* dZ, T* dX) {

  CUDA_1D_KERNEL_LOOP(i, N * D) {
    int n = i / D;
    int d = i % D;

    // hor
    for (int p = n, q = 0; q < N; q++) {
      T dz = dZ[p * N + q];
      T sub = X[p * D + d] - X[q * D + d];
      dX[n * D + d] += 2.0 * sub * (1.0) * dz;
    }

    // vec
    for (int p = 0, q = n; p < N; p++) {
      T dz = dZ[p * N + q];
      T sub = X[p * D + d] - X[q * D + d];
      dX[n * D + d] += 2.0 * sub * (-1.0) * dz;
    }
  }
}

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__global__ void PairWiseDistanceGradientKernel2(const int N, const int D,
                                                const T* X, const T* dZ,
                                                T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N * N) {
    int p = i / N;
    int q = i % N;
    T dz = dZ[p * N + q];
    for (int d = 0; d < D; d++) {
      T sub = X[p * D + d] - X[q * D + d];
      gpu_atomic_add(static_cast<T>(2.0 * sub * (1.0) * dz), dX + p * D + d);
      gpu_atomic_add(static_cast<T>(2.0 * sub * (-1.0) * dz), dX + q * D + d);
    }
  }
}
}  // namespace

template <>
bool PairWiseDistanceGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dZ = Input(1);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(dZ.dim(), 2);
  CAFFE_ENFORCE_EQ(dZ.dim32(0), X.dim32(0));
  CAFFE_ENFORCE_EQ(dZ.dim32(1), X.dim32(0));

  int N = X.dim32(0);
  int D = X.dim32(1);

  auto* dX = Output(0);
  dX->ResizeLike(X);
  math::Set<float, CUDAContext>(dX->numel(), float(0), dX->mutable_data<float>(),
                                &context_);

  // PairWiseDistanceGradientKernel
  //<< <CAFFE_GET_BLOCKS(N * D), CAFFE_CUDA_NUM_THREADS, 0,
  // context_.cuda_stream()>>>
  //(N, D, X.data<float>(), dZ.data<float>(), dX->mutable_data<float>());
  PairWiseDistanceGradientKernel2
          << <CAFFE_GET_BLOCKS(N * N), CAFFE_CUDA_NUM_THREADS, 0,
              context_.cuda_stream()>>>
      (N, D, X.data<float>(), dZ.data<float>(), dX->mutable_data<float>());

  return true;
}

REGISTER_CUDA_OPERATOR(PairWiseDistance,
                       PairWiseDistanceOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(PairWiseDistanceGradient,
                       PairWiseDistanceGradientOp<float, CUDAContext>);

}  // namespace caffe2
