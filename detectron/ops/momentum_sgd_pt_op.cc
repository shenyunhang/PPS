#include "momentum_sgd_pt_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(MomentumSGDPT, MomentumSGDPTOp<float, CPUContext>);
OPERATOR_SCHEMA(MomentumSGDPT)
    .NumInputs(3)
    .NumOutputs(2)
    .AllowInplace({{0, 0}, {1, 1}})
    .TensorInferenceFunction(
        [](const OperatorDef& /* unused */, const vector<TensorShape>& in) {
          vector<TensorShape> out(2);
          out[0] = in[0];
          out[1] = in[1];
          return out;
        })
    .SetDoc(R"DOC(

Computes a momentum SGD update for an input gradient and momentum
parameters. Concretely, given inputs (grad, m, lr) and parameters
(momentum, nesterov), computes:

    if not nesterov:
        adjusted_gradient = lr * grad + momentum * m
        return (adjusted_gradient, adjusted_gradient)
    else:
        m_new = momentum * m + lr * grad
        return ((1 + momentum) * m_new - momentum * m, m_new)

Output is (grad, momentum)

Note the difference to MomemtumSGDUpdate, which actually performs the
parameter update (and is thus faster).
)DOC");
SHOULD_NOT_DO_GRADIENT(MomentumSGDPT);

REGISTER_CPU_OPERATOR(
    MomentumSGDUpdatePT,
    MomentumSGDUpdatePTOp<float, CPUContext>);
OPERATOR_SCHEMA(MomentumSGDUpdatePT)
    .NumInputs(4)
    .NumOutputs(3)
    .AllowInplace({{0, 0}, {1, 1}, {3, 2}})
    .TensorInferenceFunction(
        [](const OperatorDef& /* unused */, const vector<TensorShape>& in) {
          vector<TensorShape> out(3);
          out[0] = in[0];
          out[1] = in[1];
          out[2] = in[3];
          return out;
        })
    .SetDoc(R"DOC(

Performs a momentum SGD update for an input gradient and momentum
parameters. Concretely, given inputs (grad, m, lr, param) and arguments
(momentum, nesterov), computes:

    if not nesterov:
        adjusted_gradient = lr * grad + momentum * m
        param = param - adjusted_gradient
        return (adjusted_gradient, adjusted_gradient, param)
    else:
        m_new = momentum * m + lr * grad
        param = param - ((1 + momentum) * m_new - momentum * m),
        return ((1 + momentum) * m_new - momentum * m, m_new, param)

Output is (grad, momentum, parameter).

Note the difference to MomentumSGDPT, which returns a new gradient
but does not perform the parameter update.

)DOC");
SHOULD_NOT_DO_GRADIENT(MomentumSGDUpdatePT);

REGISTER_CPU_OPERATOR(
    SparseMomentumSGDUpdatePT,
    SparseMomentumSGDUpdatePTOp<float, CPUContext>);
OPERATOR_SCHEMA(SparseMomentumSGDUpdatePT)
    .NumInputs(5)
    .NumOutputs(3)
    .AllowInplace({{0, 0}})
    .EnforceInplace({{1, 1}, {3, 2}})
    .TensorInferenceFunction(
        [](const OperatorDef& /* unused */, const vector<TensorShape>& in) {
          vector<TensorShape> out(3);
          out[0] = in[0];
          out[1] = in[1];
          out[2] = in[3];
          return out;
        })
    .SetDoc(R"DOC(

Performs a momentum SGD update analogous to MomentumSGDUpdatePT, but using a
GradientSlice and indices into the full param and momentum tables. Both param
and momentum should be in-place (corresponding inputs and outputs should be the
same blobs).



)DOC")
    .Input(0, "grad", "GradientSlice with gradients for updated indices.")
    .Input(1, "moment", "Momentum blob, same shape as param.")
    .Input(2, "lr", "Learning rate.")
    .Input(3, "param", "Full parameter blob.")
    .Input(
        4,
        "indices",
        "Indices (in first dimension of param) where updates are performed.")
    .Output(0, "output_grad", "Adjusted gradient.")
    .Output(1, "output_moment", "Updated momentum.")
    .Output(2, "output_param", "Updated parameter")
    .Arg("momentum", "Momentum hyperparameter.")
    .Arg("nesterov", "(boolean) Whether to use Nesterov Accelerated Gradient.");
SHOULD_NOT_DO_GRADIENT(SparseMomentumSGDUpdatePT);
}