package com.github.maximtereshchenko.snapdragon;

record FakeActivationFunction() implements ActivationFunction {

    @Override
    public Tensor apply(Tensor tensor) {
        return tensor;
    }

    @Override
    public Tensor deltas(Tensor outputs, Tensor errorSignal) {
        return errorSignal;
    }
}
