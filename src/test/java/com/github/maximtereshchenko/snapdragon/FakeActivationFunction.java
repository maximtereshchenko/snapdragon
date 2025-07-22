package com.github.maximtereshchenko.snapdragon;

record FakeActivationFunction() implements ActivationFunction {

    @Override
    public Tensor apply(Tensor tensor) {
        return tensor;
    }

    @Override
    public Tensor derivative(Tensor tensor) {
        return tensor;
    }
}
