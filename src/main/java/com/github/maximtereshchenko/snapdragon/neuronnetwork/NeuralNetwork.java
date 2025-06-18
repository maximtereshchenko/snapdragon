package com.github.maximtereshchenko.snapdragon.neuronnetwork;

import com.github.maximtereshchenko.snapdragon.matrix.Matrix;

import java.util.List;

public final class NeuralNetwork {

    private final List<Matrix> weights;
    private final List<Matrix> biases;
    private final ActivationFunction hiddenLayerActivationFunction;
    private final ActivationFunction outputLayerActivationFunction;

    public NeuralNetwork(
        List<Matrix> weights,
        List<Matrix> biases,
        ActivationFunction hiddenLayerActivationFunction,
        ActivationFunction outputLayerActivationFunction
    ) {
        this.weights = weights;
        this.biases = biases;
        this.hiddenLayerActivationFunction = hiddenLayerActivationFunction;
        this.outputLayerActivationFunction = outputLayerActivationFunction;
    }

    public Matrix outputs(Matrix inputs) {
        var outputs = inputs;
        for (var i = 0; i < weights.size(); i++) {
            outputs = activationFunction(i)
                          .apply(
                              outputs.product(weights.get(i))
                                  .combined(biases.get(i), Double::sum)
                          );
        }
        return outputs;
    }

    private ActivationFunction activationFunction(int index) {
        if (index == weights.size() - 1) {
            return outputLayerActivationFunction;
        }
        return hiddenLayerActivationFunction;
    }
}
