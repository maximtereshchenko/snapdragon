package com.github.maximtereshchenko.snapdragon.domain;

import java.util.ArrayList;
import java.util.List;

final class HiddenLayer {

    private final List<Double> biases;
    private final List<List<Double>> weights;
    private final ActivationFunction activationFunction;

    HiddenLayer(
        List<Double> biases,
        List<List<Double>> weights,
        ActivationFunction activationFunction
    ) {
        this.biases = biases;
        this.weights = weights;
        this.activationFunction = activationFunction;
    }

    NeuralNetworkSegment neuralNetworkSegment(Connectable connectable) {
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
        return new NeuralNetworkSegment(neurons, weights);
    }
}
