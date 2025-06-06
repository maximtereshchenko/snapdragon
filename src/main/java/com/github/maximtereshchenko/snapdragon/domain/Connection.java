package com.github.maximtereshchenko.snapdragon.domain;

final class Connection {

    private final Neuron neuron;
    private final double weight;

    Connection(Neuron neuron, double weight) {
        this.neuron = neuron;
        this.weight = weight;
    }

    double value(double[] inputs) {
        return neuron.value(inputs) * weight;
    }
}
