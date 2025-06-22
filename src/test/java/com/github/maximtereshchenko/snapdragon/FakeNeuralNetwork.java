package com.github.maximtereshchenko.snapdragon;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Queue;

final class FakeNeuralNetwork implements NeuralNetwork {

    private final Queue<Double> inputScales;
    private final int generation;
    private int attempts;

    private FakeNeuralNetwork(Queue<Double> inputScales, int generation, int attempts) {
        this.inputScales = inputScales;
        this.generation = generation;
        this.attempts = attempts;
    }

    public FakeNeuralNetwork(Double... inputScales) {
        this(new LinkedList<>(List.of(inputScales)), 0, 0);
    }

    public FakeNeuralNetwork(int generation, int attempts) {
        this(new LinkedList<>(), generation, attempts);
    }

    @Override
    public Matrix prediction(Matrix inputs) {
        var scale = inputScales.poll();
        if (scale == null) {
            return inputs;
        }
        return inputs.applied(value -> value * scale);
    }

    @Override
    public NeuralNetwork adjusted(
        Matrix inputs,
        Matrix labels,
        LossFunction lossFunction,
        double learningRate
    ) {
        return new FakeNeuralNetwork(inputScales, generation + 1, attempts++);
    }

    @Override
    public int hashCode() {
        return Objects.hash(inputScales, generation);
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        return object instanceof FakeNeuralNetwork that &&
                   generation == that.generation &&
                   Objects.equals(inputScales, that.inputScales);
    }

    @Override
    public String toString() {
        return "FakeNeuralNetwork{" +
                   "inputDeltas=" + inputScales +
                   ", generation=" + generation +
                   '}';
    }
}
