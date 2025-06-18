package com.github.maximtereshchenko.snapdragon.neuronnetwork.domain;

import java.util.List;

final class InputLayer implements ConnectableSegment {

    private final List<InputNeuron> neurons;

    InputLayer(List<InputNeuron> neurons) {
        this.neurons = neurons;
    }

    @Override
    public List<Connection> connections(int destinationIndex) {
        return neurons.stream()
                   .map(neuron -> neuron.connection(destinationIndex))
                   .toList();
    }
}
