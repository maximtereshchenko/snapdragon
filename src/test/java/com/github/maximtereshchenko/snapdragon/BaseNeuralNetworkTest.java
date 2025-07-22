package com.github.maximtereshchenko.snapdragon;

import java.util.List;

abstract class BaseNeuralNetworkTest {

    final NeuralNetwork neuralNetwork(List<Tensor> weights, List<Tensor> biases) {
        var activationFunction = new FakeActivationFunction();
        return new NeuralNetworkFactory()
                   .neuralNetwork(
                       weights,
                       biases,
                       activationFunction,
                       activationFunction
                   );
    }
}
