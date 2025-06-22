package com.github.maximtereshchenko.snapdragon;

import java.util.List;
import java.util.Objects;

final class FakeNeuralNetwork implements NeuralNetwork {

    private final List<Double> inputDeltas;
    private final int generation;

    private FakeNeuralNetwork(List<Double> inputDeltas, int generation) {
        this.inputDeltas = inputDeltas;
        this.generation = generation;
    }

    public FakeNeuralNetwork(Double... inputDeltas) {
        this(List.of(inputDeltas), 0);
    }

    public FakeNeuralNetwork(int generation) {
        this(List.of(), generation);
    }

    @Override
    public Matrix prediction(Matrix inputs) {
        if (inputDeltas.isEmpty()) {
            return inputs;
        }
        return inputs.applied(value -> value - inputDeltas.getFirst());
    }

    @Override
    public NeuralNetwork adjusted(
        Matrix inputs,
        Matrix labels,
        LossFunction lossFunction,
        double learningRate
    ) {
        return new FakeNeuralNetwork(
            inputDeltas.subList(Math.min(1, inputDeltas.size()), inputDeltas.size()),
            generation + 1
        );
    }

    @Override
    public int hashCode() {
        return Objects.hash(inputDeltas, generation);
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        return object instanceof FakeNeuralNetwork that &&
                   generation == that.generation &&
                   Objects.equals(inputDeltas, that.inputDeltas);
    }

    @Override
    public String toString() {
        return "FakeNeuralNetwork{" +
                   "inputDeltas=" + inputDeltas +
                   ", generation=" + generation +
                   '}';
    }
}
