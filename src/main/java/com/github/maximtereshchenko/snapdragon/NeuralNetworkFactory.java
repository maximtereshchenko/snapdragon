package com.github.maximtereshchenko.snapdragon;

import java.util.ArrayList;
import java.util.List;

final class NeuralNetworkFactory {

    NeuralNetwork neuralNetwork(
        List<Tensor> weights,
        List<Tensor> biases,
        ActivationFunction hiddenLayerActivationFunction,
        ActivationFunction outputLayerActivationFunction
    ) {
        if (weights.isEmpty() || weights.size() != biases.size()) {
            throw new IllegalArgumentException();
        }
        var inputLayer = new InputLayer(weights.getFirst().shape().getFirst());
        var hiddenLayers = new ArrayList<HiddenLayer>();
        var networkWeights = new NetworkWeights();
        Layer left = inputLayer;
        for (var i = 0; i < biases.size() - 1; i++) {
            var hiddenLayer = new HiddenLayer(
                new LayerIndex(i + 1),
                new Biases(biases.get(i)),
                hiddenLayerActivationFunction
            );
            networkWeights = networkWeights(networkWeights, left, hiddenLayer, weights.get(i));
            hiddenLayers.add(hiddenLayer);
            left = hiddenLayer;
        }
        var outputLayer = new OutputLayer(
            new LayerIndex(biases.size()),
            new Biases(biases.getLast()),
            outputLayerActivationFunction
        );
        networkWeights = networkWeights(networkWeights, left, outputLayer, weights.getLast());
        return new MultiLayerPerceptron(
            new NetworkLayers(inputLayer, hiddenLayers, outputLayer),
            networkWeights
        );
    }

    private NetworkWeights networkWeights(
        NetworkWeights networkWeights,
        Layer left,
        Layer right,
        Tensor tensor
    ) {
        var shape = tensor.shape();
        if (
            shape.size() != 2 ||
                shape.getFirst() != left.size() ||
                shape.getLast() != right.size()
        ) {
            throw new IllegalArgumentException();
        }
        return networkWeights.with(left.index(), right.index(), new Weights(tensor));
    }
}
