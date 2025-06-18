package com.github.maximtereshchenko.snapdragon.neuronnetwork.domain;

import java.util.List;

final class NeuralNetworkSegment implements ConnectableSegment {

    private final List<IncompleteNeuron> neurons;

    NeuralNetworkSegment(List<IncompleteNeuron> neurons) {
        this.neurons = neurons;
    }

    @Override
    public List<Connection> connections(int destinationIndex) {
        return neurons.stream()
                   .map(neuron -> neuron.connection(destinationIndex))
                   .toList();
    }
}
