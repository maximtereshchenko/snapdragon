package com.github.maximtereshchenko.snapdragon;

public record Sigmoid() implements ActivationFunction {

    @Override
    public Matrix apply(Matrix matrix) {
        return matrix.applied(value -> 1 / (1 + Math.pow(Math.E, -value)));
    }

    @Override
    public Matrix derivative(Matrix matrix) {
        return matrix.applied(value -> value * (1 - value));
    }
}
