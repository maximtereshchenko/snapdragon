package com.github.maximtereshchenko.snapdragon;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

final class NeuralNetworkFactory {

    NeuralNetwork neuralNetwork(
        Random random,
        int inputs,
        int hiddenLayers,
        int outputs,
        ActivationFunction hiddenLayerActivationFunction,
        ActivationFunction outputLayerActivationFunction
    ) {
        var hiddenNeurons = (inputs + outputs) * 2 / 3;
        return neuralNetwork(
            randomWeights(random, inputs, hiddenNeurons, hiddenLayers, outputs),
            randomBiases(random, hiddenNeurons, hiddenLayers, outputs),
            hiddenLayerActivationFunction,
            outputLayerActivationFunction
        );
    }

    NeuralNetwork neuralNetwork(
        List<Tensor> weights,
        List<Tensor> biases,
        ActivationFunction hiddenLayerActivationFunction,
        ActivationFunction outputLayerActivationFunction
    ) {
        if (weights.isEmpty() || weights.size() != biases.size()) {
            throw new IllegalArgumentException();
        }
        var inputLayer = new InputLayer(weights.getFirst().shape()[0]);
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

    private List<Tensor> randomBiases(
        Random random,
        int hiddenNeurons,
        int hiddenLayers,
        int outputs
    ) {
        return Stream.concat(
                Stream.generate(() ->
                                    Tensor.horizontalVector(random.doubles(hiddenNeurons).toArray())
                    )
                    .limit(hiddenLayers),
                Stream.of(Tensor.horizontalVector(random.doubles(outputs).toArray()))
            )

                   .toList();
    }

    private List<Tensor> randomWeights(
        Random random,
        int inputs,
        int hiddenNeurons,
        int hiddenLayers,
        int outputs
    ) {
        var previousSize = inputs;
        var weights = new ArrayList<Tensor>();
        for (var i = 0; i < hiddenLayers; i++) {
            weights.add(randomWeightsTensor(random, hiddenNeurons, previousSize));
            previousSize = hiddenNeurons;
        }
        weights.add(randomWeightsTensor(random, outputs, previousSize));
        return weights;
    }

    private Tensor randomWeightsTensor(Random random, int outputs, int previousSize) {
        return Tensor.matrix(
            previousSize, outputs,
            random.doubles((long) previousSize * outputs)
                .toArray()
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
            shape.length != 2 ||
                shape[0] != left.size() ||
                shape[shape.length - 1] != right.size()
        ) {
            throw new IllegalArgumentException();
        }
        return networkWeights.with(left.index(), right.index(), new Weights(tensor));
    }
}
