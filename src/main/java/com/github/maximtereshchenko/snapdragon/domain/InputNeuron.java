package com.github.maximtereshchenko.snapdragon.domain;

import java.util.List;

final class InputNeuron implements ConnectableNeuron {

    private final int index;
    private final List<Double> weights;

    InputNeuron(int index, List<Double> weights) {
        this.index = index;
        this.weights = weights;
    }

    @Override
    public Connection connection(int destinationIndex) {
        return new Connection(
            new PredictingInputNeuron(index),
            weights.get(destinationIndex)
        );
    }
}
