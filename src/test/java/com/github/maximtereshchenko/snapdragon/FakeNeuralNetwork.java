package com.github.maximtereshchenko.snapdragon;

final class FakeNeuralNetwork implements NeuralNetwork {

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
        return this;
    }
}
