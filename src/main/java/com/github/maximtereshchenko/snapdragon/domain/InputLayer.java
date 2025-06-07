package com.github.maximtereshchenko.snapdragon.domain;

import java.util.ArrayList;
import java.util.List;

final class InputLayer implements Connectable {

    private final int inputs;
    private final List<List<Double>> weights;

    InputLayer(int inputs, List<List<Double>> weights) {
        this.inputs = inputs;
        this.weights = weights;
    }

    @Override
    public List<Connection> connections(int destinationIndex) {
        var connections = new ArrayList<Connection>();
        for (var i = 0; i < inputs; i++) {
            connections.add(
                new Connection(
                    new InputNeuron(i),
                    weights.get(i).get(destinationIndex)
                )
            );
        }
        return connections;
    }
}
