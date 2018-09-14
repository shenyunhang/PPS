#include "batch_hard_op.h"
#include <limits>

namespace caffe2 {

namespace {}

template <>
bool BatchHardOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& L = Input(1);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(L.dim(), 1);
  CAFFE_ENFORCE_EQ(X.dim32(0), X.dim32(1));
  CAFFE_ENFORCE_EQ(X.dim32(0), L.dim32(0));

  int N = X.dim32(0);

  auto* AP = Output(0);
  auto* AN = Output(1);
  AP->Resize(N);
  AN->Resize(N);

  const float* Xdata = X.data<float>();
  const int* Ldata = L.data<int>();
  auto* APdata = AP->mutable_data<float>();
  auto* ANdata = AN->mutable_data<float>();

  for (int a = 0; a < N; a++) {
    int la = Ldata[a];

    float dist_ap = 0;
    for (int p = 0; p < N; p++) {
      int lp = Ldata[p];
      if (la != lp) {
        continue;
      }
      if (dist_ap < Xdata[a * N + p]) {
        dist_ap = Xdata[a * N + p];
      }
    }
    APdata[a] = dist_ap;

    float dist_an = std::numeric_limits<float>::max();
    for (int n = 0; n < N; n++) {
      int ln = Ldata[n];
      if (la == ln) {
        continue;
      }
      if (dist_an > Xdata[a * N + n]) {
        dist_an = Xdata[a * N + n];
      }
    }
    ANdata[a] = dist_an;
  }

  return true;
}

template <>
bool BatchHardGradientOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& L = Input(1);
  const auto& dAP = Input(2);
  const auto& dAN = Input(3);

  CAFFE_ENFORCE_EQ(X.ndim(), 2);
  CAFFE_ENFORCE_EQ(L.ndim(), 1);
  CAFFE_ENFORCE_EQ(dAP.ndim(), 1);
  CAFFE_ENFORCE_EQ(dAN.ndim(), 1);
  CAFFE_ENFORCE_EQ(X.dim32(0), X.dim32(1));
  CAFFE_ENFORCE_EQ(X.dim32(0), L.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(0), dAP.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(0), dAN.dim32(0));

  int N = X.dim32(0);

  auto* dX = Output(0);
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(dX->numel(), 0.f, dX->mutable_data<float>(),
                               &context_);

  const float* Xdata = X.data<float>();
  const int* Ldata = L.data<int>();
  const float* dAPdata = dAP.data<float>();
  const float* dANdata = dAN.data<float>();
  float* dXdata = dX->mutable_data<float>();

  for (int a = 0; a < N; a++) {
    int la = Ldata[a];

    float dist_ap = 0;
    int idx_p = -1;
    for (int p = 0; p < N; p++) {
      int lp = Ldata[p];
      if (la != lp) {
        continue;
      }
      if (dist_ap < Xdata[a * N + p]) {
        dist_ap = Xdata[a * N + p];
        idx_p = p;
      }
    }
    dXdata[a * N + idx_p] = dAPdata[a];

    float dist_an = std::numeric_limits<float>::max();
    int idx_n = -1;
    for (int n = 0; n < N; n++) {
      int ln = Ldata[n];
      if (la == ln) {
        continue;
      }
      if (dist_an > Xdata[a * N + n]) {
        dist_an = Xdata[a * N + n];
        idx_n = n;
      }
    }
    dXdata[a * N + idx_n] = dANdata[a];
  }

  return true;
}

REGISTER_CPU_OPERATOR(BatchHard, BatchHardOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(BatchHardGradient,
                      BatchHardGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(BatchHard)
    .NumInputs(2)
    .NumOutputs(2)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
)DOC")
    .Input(0, "X", "2D input tensor")
    .Input(1, "L", "1D input tensor")
    .Output(0, "AP", "1D output tensor")
    .Output(1, "AN", "1D output tensor");

OPERATOR_SCHEMA(BatchHardGradient).NumInputs(4).NumOutputs(1);

class GetBatchHardGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef("BatchHardGradient", "",
                             vector<string>{I(0), I(1), GO(0), GO(1)},
                             vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(BatchHard, GetBatchHardGradient);

}  // namespace caffe2
