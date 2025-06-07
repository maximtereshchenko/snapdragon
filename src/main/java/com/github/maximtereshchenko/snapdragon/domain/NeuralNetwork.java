package com.github.maximtereshchenko.snapdragon.domain;

import java.util.List;

public final class NeuralNetwork {

    private final List<StaticNeuron> outputs;

    NeuralNetwork(List<StaticNeuron> outputs) {
        this.outputs = outputs;
    }

    public double[] prediction(double[] inputs) {
        return outputs.stream()
                   .mapToDouble(output -> output.value(inputs))
                   .toArray();
    }
}
