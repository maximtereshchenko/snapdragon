package com.github.maximtereshchenko.snapdragon.neuronnetwork.domain;

final class Connection {

    private final PredictingNeuron neuron;
    private final double weight;

    Connection(PredictingNeuron neuron, double weight) {
        this.neuron = neuron;
        this.weight = weight;
    }

    double value(double[] inputs) {
        return neuron.prediction(inputs) * weight;
    }
}
