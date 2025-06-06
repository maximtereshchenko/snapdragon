package com.github.maximtereshchenko.snapdragon.domain;

import java.util.List;

final class StaticNeuron implements Neuron {

    private final List<Connection> connections;
    private final double bias;
    private final ActivationFunction activationFunction;

    StaticNeuron(List<Connection> connections, double bias, ActivationFunction activationFunction) {
        this.connections = connections;
        this.bias = bias;
        this.activationFunction = activationFunction;
    }

    @Override
    public double value(double[] inputs) {
        return activationFunction.apply(
            connections.stream()
                .mapToDouble(connection -> connection.value(inputs))
                .sum()
                + bias
        );
    }
}
