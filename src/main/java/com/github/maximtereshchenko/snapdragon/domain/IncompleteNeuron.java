package com.github.maximtereshchenko.snapdragon.domain;

import java.util.List;

final class IncompleteNeuron implements ConnectableNeuron {

    private final PredictingStaticNeuron neuron;
    private final List<Double> weights;

    IncompleteNeuron(PredictingStaticNeuron neuron, List<Double> weights) {
        this.neuron = neuron;
        this.weights = weights;
    }

    @Override
    public Connection connection(int destinationIndex) {
        return new Connection(neuron, weights.get(destinationIndex));
    }
}
