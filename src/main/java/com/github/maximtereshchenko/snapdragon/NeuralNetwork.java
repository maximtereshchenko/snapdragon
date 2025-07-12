package com.github.maximtereshchenko.snapdragon;

interface NeuralNetwork {

    Outputs outputs(Inputs inputs);

    NeuralNetwork calibrated(
        Inputs inputs,
        Labels labels,
        LossFunction lossFunction,
        LearningRate learningRate
    );
}
