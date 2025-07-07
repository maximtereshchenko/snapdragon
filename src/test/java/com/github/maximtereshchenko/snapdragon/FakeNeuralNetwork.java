package com.github.maximtereshchenko.snapdragon;

import java.util.Objects;

final class FakeNeuralNetwork implements NeuralNetwork {

    private final int generation;
    private int attempts;

    FakeNeuralNetwork(int generation, int attempts) {
        this.generation = generation;
        this.attempts = attempts;
    }

    FakeNeuralNetwork() {
        this(0, 0);
    }

    @Override
    public Matrix prediction(Matrix inputs) {
        return inputs;
    }

    @Override
    public NeuralNetwork adjusted(
        Matrix inputs,
        Matrix labels,
        LossFunction lossFunction,
        double learningRate
    ) {
        return new FakeNeuralNetwork(generation + 1, attempts++);
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

    @Override
    public String toString() {
        return "FakeNeuralNetwork{" +
                   "generation=" + generation +
                   ", attempts=" + attempts +
                   '}';
    }
}
