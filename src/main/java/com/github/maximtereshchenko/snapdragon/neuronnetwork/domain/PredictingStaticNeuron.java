package com.github.maximtereshchenko.snapdragon.neuronnetwork.domain;

import java.util.List;

final class PredictingStaticNeuron implements PredictingNeuron {

    private final List<Connection> connections;
    private final double bias;
    private final ActivationFunction activationFunction;

    PredictingStaticNeuron(
        List<Connection> connections,
        double bias,
        ActivationFunction activationFunction
    ) {
        this.connections = connections;
        this.bias = bias;
        this.activationFunction = activationFunction;
    }

    @Override
    public double prediction(double[] inputs) {
        return activationFunction.apply(
            connections.stream()
                .mapToDouble(connection -> connection.value(inputs))
                .sum()
                + bias
        );
    }
}
