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
    public Tensor loss(Tensor outputs, Tensor labels) {
        return Tensor.verticalVector(losses.remove())
                   .broadcasted(labels.shape()[0], 1);
    }

    @Override
    public Tensor derivative(Tensor outputs, Tensor labels) {
        return outputs;
    }
}
