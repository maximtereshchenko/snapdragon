package com.github.maximtereshchenko.snapdragon;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

final class FakeLossFunction implements LossFunction {

    private final Queue<Double> losses;

    private FakeLossFunction(Queue<Double> losses) {
        this.losses = losses;
    }

    FakeLossFunction(Double... losses) {
        this(new LinkedList<>(List.of(losses)));
    }

    @Override
    public Matrix loss(Matrix outputs, Matrix labels) {
        return Matrix.verticalVector(losses.remove())
                   .broadcasted(labels.rows(), 1);
    }

    @Override
    public Matrix derivative(Matrix outputs, Matrix labels) {
        return outputs;
    }
}
