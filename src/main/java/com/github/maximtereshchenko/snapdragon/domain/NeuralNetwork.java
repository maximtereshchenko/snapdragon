package com.github.maximtereshchenko.snapdragon.domain;

import com.github.maximtereshchenko.snapdragon.api.NeuralNetworkConfiguration;
import java.util.ArrayList;
import java.util.List;

public final class NeuralNetwork {

    private final List<StaticNeuron> outputs;

    private NeuralNetwork(List<StaticNeuron> outputs) {
        this.outputs = outputs;
    }

    public static NeuralNetwork from(NeuralNetworkConfiguration configuration) {
        return new NeuralNetwork(outputs(configuration, new Sigmoid()));
    }

    private static List<StaticNeuron> outputs(
        NeuralNetworkConfiguration configuration,
        ActivationFunction activationFunction
    ) {
        var outputs = new ArrayList<StaticNeuron>();
        var biases = configuration.outputLayerConfiguration().biases();
        for (var i = 0; i < biases.size(); i++) {
            outputs.add(
                new StaticNeuron(
                    connections(configuration, i),
                    biases.get(i),
                    activationFunction
                )
            );
        }
        return outputs;
    }

    private static List<Connection> connections(NeuralNetworkConfiguration configuration, int destinationIndex) {
        var connections = new ArrayList<Connection>();
        var inputs = configuration.inputLayerConfiguration().inputs();
        for (var index = 0; index < inputs; index++) {
            var neuron = new InputNeuron(index);
            connections.add(
                new Connection(
                    neuron,
                    configuration.inputLayerConfiguration().weights().get(destinationIndex + index)
                )
            );
        }
        return connections;
    }

    public double[] prediction(double[] inputs) {
        return outputs.stream()
            .mapToDouble(output -> output.value(inputs))
            .toArray();
    }
}
