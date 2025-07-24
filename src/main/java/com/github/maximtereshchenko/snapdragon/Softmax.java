package com.github.maximtereshchenko.snapdragon;

import java.util.List;

record Softmax() implements ActivationFunction {

    @Override
    public Tensor apply(Tensor tensor) {
        var shape = tensor.shape();
        var exponents = Tensor.from(shape, index -> Math.exp(tensor.value(index)));
        return exponents.quotient(
            exponents.contracted(
                    Tensor.horizontalVector(1)
                        .broadcasted(shape.getLast(), 1)
                )
                .broadcasted(shape)
        );
    }

    @Override
    public Tensor deltas(Tensor outputs, Tensor errorSignal) {
        var shape = outputs.shape();
        var batchedJacobianMatrices = Tensor.from(
            List.of(shape.getFirst(), shape.getLast(), shape.getLast()),
            index -> jacobian(index, outputs)
        );
        return errorSignal.batchContracted(batchedJacobianMatrices);
    }

    private double jacobian(List<Integer> index, Tensor outputs) {
        var batch = index.getFirst();
        var row = index.get(1);
        var column = index.getLast();
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
