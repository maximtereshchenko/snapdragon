package com.github.maximtereshchenko.snapdragon.domain;

import com.github.maximtereshchenko.snapdragon.api.HiddenLayerConfiguration;
import com.github.maximtereshchenko.snapdragon.api.InputLayerConfiguration;
import com.github.maximtereshchenko.snapdragon.api.NeuralNetworkConfiguration;
import com.github.maximtereshchenko.snapdragon.api.OutputLayerConfiguration;

import java.util.ArrayList;
import java.util.List;

public final class NeuralNetworkFactory {

    public NeuralNetwork neuralNetwork(NeuralNetworkConfiguration configuration) {
        var activationFunction = new Sigmoid();
        return neuralNetwork(
            inputLayer(configuration.inputLayerConfiguration()),
            hiddenLayers(configuration.hiddenLayerConfigurations(), activationFunction),
            outputLayer(configuration.outputLayerConfiguration(), activationFunction)
        );
    }

    private OutputLayer outputLayer(
        OutputLayerConfiguration outputLayerConfiguration,
        ActivationFunction activationFunction
    ) {
        return new OutputLayer(
            outputLayerConfiguration.outputNeuronConfigurations()
                .stream()
                .map(outputNeuronConfiguration ->
                         new OutputLayerNeuron(
                             outputNeuronConfiguration.bias(),
                             activationFunction
                         )
                )
                .toList()
        );
    }

    private List<HiddenLayer> hiddenLayers(
        List<HiddenLayerConfiguration> configurations,
        ActivationFunction activationFunction
    ) {
        return configurations.stream()
                   .map(hiddenLayerConfiguration ->
                            hiddenLayer(hiddenLayerConfiguration, activationFunction)
                   )
                   .toList();
    }

    private HiddenLayer hiddenLayer(
        HiddenLayerConfiguration hiddenLayerConfiguration,
        ActivationFunction activationFunction
    ) {
        return new HiddenLayer(
            hiddenLayerConfiguration.hiddenNeuronConfigurations()
                .stream()
                .map(hiddenNeuronConfiguration ->
                         new HiddenLayerNeuron(
                             hiddenNeuronConfiguration.bias(),
                             hiddenNeuronConfiguration.weights(),
                             activationFunction
                         )
                )
                .toList()
        );
    }

    private InputLayer inputLayer(InputLayerConfiguration configuration) {
        var inputNeuronConfigurations = configuration.inputNeuronConfigurations();
        var inputNeurons = new ArrayList<InputNeuron>();
        for (var i = 0; i < inputNeuronConfigurations.size(); i++) {
            inputNeurons.add(
                new InputNeuron(i, inputNeuronConfigurations.get(i).weights())
            );
        }
        return new InputLayer(inputNeurons);
    }

    private NeuralNetwork neuralNetwork(
        InputLayer inputLayer,
        List<HiddenLayer> hiddenLayers,
        OutputLayer outputLayer
    ) {
        ConnectableSegment connectable = inputLayer;
        for (var hiddenLayer : hiddenLayers) {
            connectable = hiddenLayer.neuralNetworkSegment(connectable);
        }
        return outputLayer.neuralNetwork(connectable);
    }
}
