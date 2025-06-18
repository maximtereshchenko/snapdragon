package com.github.maximtereshchenko.snapdragon.neuronnetwork.domain;

import java.util.List;

final class HiddenLayerNeuron {

    private final double bias;
    private final List<Double> weights;
    private final ActivationFunction activationFunction;

    HiddenLayerNeuron(double bias, List<Double> weights, ActivationFunction activationFunction) {
        this.bias = bias;
        this.weights = weights;
        this.activationFunction = activationFunction;
    }

    IncompleteNeuron incompleteNeuron(List<Connection> connections) {
        return new IncompleteNeuron(
            new PredictingStaticNeuron(connections, bias, activationFunction),
            weights
        );
    }
}
