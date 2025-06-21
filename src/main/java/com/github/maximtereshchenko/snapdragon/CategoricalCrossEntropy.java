package com.github.maximtereshchenko.snapdragon;

public record CategoricalCrossEntropy() implements LossFunction {

    @Override
    public Matrix derivative(Matrix outputs, Matrix labels) {
        return labels.combined(outputs, (label, output) -> -label / output);
    }
}
