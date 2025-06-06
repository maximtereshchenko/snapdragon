package com.github.maximtereshchenko.snapdragon.domain;

import com.github.maximtereshchenko.snapdragon.api.NeuralNetworkConfiguration;
import java.util.ArrayList;
import java.util.List;

public final class NeuralNetwork {

    private final StaticNeuron outputNeuron;

    private NeuralNetwork(StaticNeuron outputNeuron) {
        this.outputNeuron = outputNeuron;
    }

    public static NeuralNetwork from(NeuralNetworkConfiguration configuration) {
        return new NeuralNetwork(
            new StaticNeuron(
                connections(configuration),
                configuration.outputLayerConfiguration().biases().getFirst(),
                new Sigmoid()
            )
        );
    }

    private static List<Connection> connections(NeuralNetworkConfiguration configuration) {
        var connections = new ArrayList<Connection>();
        for (var index = 0; index < configuration.inputLayerConfiguration().inputs(); index++) {
            var neuron = new InputNeuron(index);
            for (
                var destinationIndex = 0;
                destinationIndex < configuration.outputLayerConfiguration().biases().size();
                destinationIndex++
            ) {
                connections.add(
                    new Connection(
                        neuron,
                        configuration.inputLayerConfiguration().weights().get(destinationIndex)
                    )
                );
            }
        }
        return connections;
    }

    public double[] prediction(double[] inputs) {
        return new double[]{outputNeuron.value(inputs)};
    }
}
