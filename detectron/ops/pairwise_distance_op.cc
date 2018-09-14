#include "pairwise_distance_op.h"

namespace caffe2 {

OPERATOR_SCHEMA(PairWiseDistance)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
)DOC")
    .Input(0, "X", "2D input tensor")
    .Output(0, "Z", "2D output tensor");

OPERATOR_SCHEMA(PairWiseDistanceGradient).NumInputs(2).NumOutputs(1);

class GetPairWiseDistanceGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef("PairWiseDistanceGradient", "",
                             vector<string>{I(0), GO(0)},
                             vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(PairWiseDistance, GetPairWiseDistanceGradient);

}  // namespace caffe2
