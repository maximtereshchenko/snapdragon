package com.github.maximtereshchenko.snapdragon.neuronnetwork.domain;

import java.util.ArrayList;
import java.util.List;

final class OutputLayer {

    private final List<OutputLayerNeuron> neurons;

    OutputLayer(List<OutputLayerNeuron> neurons) {
        this.neurons = neurons;
    }

    NeuralNetwork neuralNetwork(ConnectableSegment connectable) {
        var predictingStaticNeurons = new ArrayList<PredictingStaticNeuron>();
        for (var i = 0; i < neurons.size(); i++) {
            predictingStaticNeurons.add(
                neurons.get(i)
                    .predictingStaticNeuron(connectable.connections(i))
            );
        }
        return new NeuralNetwork(predictingStaticNeurons);
    }
}
