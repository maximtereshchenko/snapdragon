package com.github.maximtereshchenko.snapdragon.domain;

import java.util.ArrayList;
import java.util.List;

final class OutputLayer {

    private final List<Double> biases;
    private final ActivationFunction activationFunction;

    OutputLayer(List<Double> biases, ActivationFunction activationFunction) {
        this.biases = biases;
        this.activationFunction = activationFunction;
    }

    NeuralNetwork neuralNetwork(Connectable connectable) {
        var neurons = new ArrayList<StaticNeuron>();
        for (var i = 0; i < biases.size(); i++) {
            neurons.add(
                new StaticNeuron(
                    connectable.connections(i),
                    biases.get(i),
                    activationFunction
                )
            );
        }
        return new NeuralNetwork(neurons);
    }
}
