package com.github.maximtereshchenko.snapdragon;

public record CategoricalCrossEntropy() implements LossFunction {

    @Override
    public Matrix loss(Matrix outputs, Matrix labels) {
        var loss = new double[outputs.rows()];
        for (var row = 0; row < outputs.rows(); row++) {
            for (var column = 0; column < outputs.columns(); column++) {
                loss[row] += labels.value(row, column) * Math.log(outputs.value(row, column));
            }
            loss[row] /= -outputs.columns();
        }
        return Matrix.verticalVector(loss);
    }

    @Override
    public Matrix derivative(Matrix outputs, Matrix labels) {
        return labels.combined(outputs, (label, output) -> -label / output);
    }
}
