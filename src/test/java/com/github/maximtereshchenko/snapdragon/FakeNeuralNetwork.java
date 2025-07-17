package com.github.maximtereshchenko.snapdragon;

import java.util.Objects;

final class FakeNeuralNetwork implements NeuralNetwork {

    private final int generation;

    FakeNeuralNetwork(int generation) {
        this.generation = generation;
    }

    FakeNeuralNetwork() {
        this(0);
    }

    @Override
    public Outputs outputs(Inputs inputs) {
        return new Outputs(inputs.matrix());
    }

    @Override
    public NeuralNetwork calibrated(
        Inputs inputs,
        Labels labels,
        LossFunction lossFunction,
        LearningRate learningRate
    ) {
        return this;
    }

    @Override
    public int hashCode() {
        return Objects.hash(generation);
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        return object instanceof FakeNeuralNetwork that &&
                   generation == that.generation;
    }
}
