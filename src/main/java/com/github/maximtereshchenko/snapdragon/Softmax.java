package com.github.maximtereshchenko.snapdragon;

record Softmax() implements ActivationFunction {

    @Override
    public Tensor apply(Tensor tensor) {
        var shape = tensor.shape();
        var exponents = Tensor.from(shape, index -> Math.exp(tensor.value(index)));
        return exponents.quotient(
            exponents.contracted(
                    Tensor.horizontalVector(1)
                        .broadcasted(shape[shape.length - 1], 1)
                )
                .broadcasted(shape)
        );
    }

    @Override
    public Tensor deltas(Tensor outputs, Tensor errorSignal) {
        var shape = outputs.shape();
        var batchedJacobianMatrices = Tensor.from(
            new int[]{shape[0], shape[shape.length - 1], shape[shape.length - 1]},
            index -> jacobian(index, outputs)
        );
        return errorSignal.batchContracted(batchedJacobianMatrices);
    }

    private double jacobian(int[] index, Tensor outputs) {
        var batch = index[0];
        var row = index[1];
        var column = index[2];
        return outputs.value(batch, row) *
                   (kroneckerDelta(row, column) - outputs.value(batch, column));
    }

    private int kroneckerDelta(int row, int column) {
        if (row == column) {
            return 1;
        }
        return 0;
    }
}
