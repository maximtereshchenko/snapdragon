package com.github.maximtereshchenko.snapdragon;

final class ReLU implements ActivationFunction {

    @Override
    public Matrix apply(Matrix matrix) {
        return matrix.applied(value -> Math.max(0, value));
    }

    @Override
    public Matrix derivative(Matrix matrix) {
        return matrix.applied(this::derivative);
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
