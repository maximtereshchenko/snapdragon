package com.github.maximtereshchenko.snapdragon;

public interface NeuralNetwork {

    Matrix prediction(Matrix inputs);

    NeuralNetwork adjusted(
        Matrix inputs,
        Matrix labels,
        LossFunction lossFunction,
        double learningRate
    );
}
