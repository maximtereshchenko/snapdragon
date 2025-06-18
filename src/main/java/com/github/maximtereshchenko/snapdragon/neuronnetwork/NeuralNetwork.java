package com.github.maximtereshchenko.snapdragon.neuronnetwork;

import com.github.maximtereshchenko.snapdragon.matrix.Matrix;

import java.util.List;

public final class NeuralNetwork {

    private final List<Matrix> weights;
    private final List<Matrix> biases;
    private final ActivationFunction hiddenLayerActivationFunction;
    private final ActivationFunction outputLayerActivationFunction;

    private NeuralNetwork(
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

    public static NeuralNetwork from(
        List<Matrix> weights,
        List<Matrix> biases,
        ActivationFunction hiddenLayerActivationFunction,
        ActivationFunction outputLayerActivationFunction
    ) {
        if (weights.isEmpty() || weights.size() != biases.size()) {
            throw new IllegalArgumentException();
        }
        for (var i = 0; i < weights.size(); i++) {
            var weightMatrix = weights.get(i);
            if (
                weightMatrix.columns() != biases.get(i).columns() ||
                    i > 0 && weightMatrix.rows() != biases.get(i - 1).columns()
            ) {
                throw new IllegalArgumentException();
            }
        }
        return new NeuralNetwork(
            weights,
            biases,
            hiddenLayerActivationFunction,
            outputLayerActivationFunction
        );
    }

    public Matrix outputs(Matrix inputs) {
        if (weights.getFirst().rows() != inputs.columns()) {
            throw new IllegalArgumentException();
        }
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
