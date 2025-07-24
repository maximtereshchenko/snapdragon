package com.github.maximtereshchenko.snapdragon;

record ReLU() implements ActivationFunction {

    @Override
    public Tensor apply(Tensor tensor) {
        return Tensor.from(tensor.shape(), index -> Math.max(0, tensor.value(index)));
    }

    @Override
    public Tensor derivative(Tensor outputs, Tensor errorSignal) {
        return errorSignal.product(
            Tensor.from(outputs.shape(), index -> derivative(outputs.value(index)))
        );
    }

    private double derivative(double value) {
        if (value < 0) {
            return 0;
        }
        if (value > 0) {
            return 1;
        }
        return 0;
    }
}
