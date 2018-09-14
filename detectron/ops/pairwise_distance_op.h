#ifndef CAFFE2_OPERATORS_PAIRWISE_DISTANCE_OP_H_
#define CAFFE2_OPERATORS_PAIRWISE_DISTANCE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class PairWiseDistanceOp : public Operator<Context> {
 public:
  PairWiseDistanceOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  // Input: X; Output: Distance
};

template <typename T, class Context>
class PairWiseDistanceGradientOp final : public Operator<Context> {
 public:
  PairWiseDistanceGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  // Input: X, dDistance; Output: dX
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_PAIRWISE_DISTANCE_OP_H_
