package com.github.maximtereshchenko.snapdragon;

record FakeActivationFunction() implements ActivationFunction {

    @Override
    public Matrix apply(Matrix matrix) {
        return matrix;
    }

    @Override
    public Matrix derivative(Matrix matrix) {
        return matrix;
    }
}
