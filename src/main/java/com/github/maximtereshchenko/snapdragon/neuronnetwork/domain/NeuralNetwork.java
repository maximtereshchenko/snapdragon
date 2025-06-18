package com.github.maximtereshchenko.snapdragon.neuronnetwork.domain;

import java.util.List;

public final class NeuralNetwork {

    private final List<PredictingStaticNeuron> outputs;

    NeuralNetwork(List<PredictingStaticNeuron> outputs) {
        this.outputs = outputs;
    }

    public double[] prediction(double[] inputs) {
        return outputs.stream()
                   .mapToDouble(output -> output.prediction(inputs))
                   .toArray();
    }
}
