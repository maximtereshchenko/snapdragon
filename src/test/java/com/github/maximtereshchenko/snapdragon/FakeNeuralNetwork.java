package com.github.maximtereshchenko.snapdragon;

record FakeNeuralNetwork() implements NeuralNetwork {

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
}
