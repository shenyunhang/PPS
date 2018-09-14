#ifndef CAFFE2_OPERATORS_BATCH_HARD_OP_H_
#define CAFFE2_OPERATORS_BATCH_HARD_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class BatchHardOp : public Operator<Context> {
 public:
  BatchHardOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
};

template <typename T, class Context>
class BatchHardGradientOp final : public Operator<Context> {
 public:
  BatchHardGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_BATCH_HARD_OP_H_
