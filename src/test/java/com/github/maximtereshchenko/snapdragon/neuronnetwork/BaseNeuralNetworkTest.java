package com.github.maximtereshchenko.snapdragon.neuronnetwork;

import com.github.maximtereshchenko.snapdragon.matrix.Matrix;

import java.util.List;

abstract class BaseNeuralNetworkTest {

    final NeuralNetwork neuralNetwork(List<Matrix> weights, List<Matrix> biases) {
        return NeuralNetwork.from(weights, biases, new Sigmoid(), new Sigmoid());
    }

    final double sigmoid(double value) {
        return 1 / (1 + Math.pow(Math.E, -value));
    }
}
