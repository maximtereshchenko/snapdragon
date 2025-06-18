package com.github.maximtereshchenko.snapdragon.neuronnetwork.domain;

import java.util.ArrayList;
import java.util.List;

final class HiddenLayer {

    private final List<HiddenLayerNeuron> neurons;

    HiddenLayer(List<HiddenLayerNeuron> neurons) {
        this.neurons = neurons;
    }

    NeuralNetworkSegment neuralNetworkSegment(ConnectableSegment connectable) {
        var incompleteNeurons = new ArrayList<IncompleteNeuron>();
        for (var i = 0; i < neurons.size(); i++) {
            incompleteNeurons.add(
                neurons.get(i)
                    .incompleteNeuron(connectable.connections(i))
            );
        }
        return new NeuralNetworkSegment(incompleteNeurons);
    }
}
