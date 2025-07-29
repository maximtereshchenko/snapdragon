package com.github.maximtereshchenko.snapdragon;

record Sigmoid() implements ActivationFunction {

    @Override
    public Tensor apply(Tensor tensor) {
        return Tensor.from(tensor.shape(), index -> 1 / (1 + Math.exp(-tensor.value(index))));
    }

    @Override
    public Tensor deltas(Tensor outputs, Tensor errorSignal) {
        return errorSignal.product(
            Tensor.from(outputs.shape(), index -> derivative(outputs, index))
        );
    }

    private double derivative(Tensor outputs, int[] index) {
        var output = outputs.value(index);
        return output * (1 - output);
    }
}
