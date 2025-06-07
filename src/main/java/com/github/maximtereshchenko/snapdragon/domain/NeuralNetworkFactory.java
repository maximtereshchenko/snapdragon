package com.github.maximtereshchenko.snapdragon.domain;

import com.github.maximtereshchenko.snapdragon.api.NeuralNetworkConfiguration;

import java.util.List;

public final class NeuralNetworkFactory {

    public NeuralNetwork neuralNetwork(NeuralNetworkConfiguration configuration) {
        var activationFunction = new Sigmoid();
        return neuralNetwork(
            new InputLayer(
                configuration.inputLayerConfiguration().inputs(),
                configuration.inputLayerConfiguration().weights()
            ),
            configuration.hiddenLayerConfigurations()
                .stream()
                .map(hiddenLayerConfiguration ->
                         new HiddenLayer(
                             hiddenLayerConfiguration.biases(),
                             hiddenLayerConfiguration.weights(),
                             activationFunction
                         )
                )
                .toList(),
            new OutputLayer(
                configuration.outputLayerConfiguration().biases(),
                activationFunction
            )
        );
    }

    private NeuralNetwork neuralNetwork(
        InputLayer inputLayer,
        List<HiddenLayer> hiddenLayers,
        OutputLayer outputLayer
    ) {
        Connectable connectable = inputLayer;
        for (var hiddenLayer : hiddenLayers) {
            connectable = hiddenLayer.neuralNetworkSegment(connectable);
        }
        return outputLayer.neuralNetwork(connectable);
    }
}
