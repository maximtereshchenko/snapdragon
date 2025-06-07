package com.github.maximtereshchenko.snapdragon.domain;

import java.util.ArrayList;
import java.util.List;

final class NeuralNetworkSegment implements Connectable {

    private final List<StaticNeuron> neurons;
    private final List<List<Double>> weights;

    NeuralNetworkSegment(List<StaticNeuron> neurons, List<List<Double>> weights) {
        this.neurons = neurons;
        this.weights = weights;
    }

    @Override
    public List<Connection> connections(int destinationIndex) {
        var connections = new ArrayList<Connection>();
        for (var i = 0; i < neurons.size(); i++) {
            connections.add(
                new Connection(
                    neurons.get(i),
                    weights.get(i).get(destinationIndex)
                )
            );
        }
        return connections;
    }
}
