package com.github.maximtereshchenko.snapdragon.neuronnetwork.domain;

import java.util.List;

final class OutputLayerNeuron {

    private final double bias;
    private final ActivationFunction activationFunction;

    OutputLayerNeuron(double bias, ActivationFunction activationFunction) {
        this.bias = bias;
        this.activationFunction = activationFunction;
    }

    PredictingStaticNeuron predictingStaticNeuron(List<Connection> connections) {
        return new PredictingStaticNeuron(connections, bias, activationFunction);
    }
}
