package com.github.maximtereshchenko.snapdragon;

import java.util.List;
import java.util.Random;

abstract class BaseNeuralNetworkTest {

    private final NeuralNetworkFactory factory = new NeuralNetworkFactory();
    private final ActivationFunction activationFunction = new FakeActivationFunction();

    final NeuralNetwork neuralNetwork(List<Tensor> weights, List<Tensor> biases) {
        return factory.neuralNetwork(
            weights,
            biases,
            activationFunction,
            activationFunction
        );
    }

    final NeuralNetwork random(int inputs, int hiddenLayers, int outputs) {
        return factory.neuralNetwork(
            new Random(0),
            inputs,
            hiddenLayers,
            outputs,
            activationFunction,
            activationFunction
        );
    }
}
