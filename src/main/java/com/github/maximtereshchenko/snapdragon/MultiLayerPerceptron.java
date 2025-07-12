package com.github.maximtereshchenko.snapdragon;

import java.util.Objects;

final class MultiLayerPerceptron implements NeuralNetwork {

    private final NetworkLayers networkLayers;
    private final NetworkWeights networkWeights;

    MultiLayerPerceptron(NetworkLayers networkLayers, NetworkWeights networkWeights) {
        this.networkLayers = networkLayers;
        this.networkWeights = networkWeights;
    }

    @Override
    public Outputs outputs(Inputs inputs) {
        return networkOutputs(inputs).element(networkLayers.outputLayer().index());
    }

    @Override
    public NeuralNetwork calibrated(
        Inputs inputs,
        Labels labels,
        LossFunction lossFunction,
        LearningRate learningRate
    ) {
        var outputs = networkOutputs(inputs);
        var deltas = deltas(outputs, lossFunction, labels);
        return new MultiLayerPerceptron(
            networkLayers.calibrated(deltas, learningRate),
            networkWeights.calibrated(outputs, deltas, learningRate)
        );
    }

    @Override
    public int hashCode() {
        return Objects.hash(networkLayers, networkWeights);
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        return object instanceof MultiLayerPerceptron that &&
                   Objects.equals(networkLayers, that.networkLayers) &&
                   Objects.equals(networkWeights, that.networkWeights);
    }

    @Override
    public String toString() {
        return "MultiLayerPerceptron{" +
                   "networkLayers=" + networkLayers +
                   ", networkWeights=" + networkWeights +
                   '}';
    }

    private LayerMap<Outputs> networkOutputs(Inputs inputs) {
        return networkLayers.forwardPropagationSegments(networkWeights)
                   .stream()
                   .reduce(
                       new LayerMap<>(
                           networkLayers.inputLayer().index(),
                           networkLayers.inputLayer().outputs(inputs)
                       ),
                       (outputs, segment) -> outputs.with(
                           segment.participant().index(),
                           segment.outputs(outputs)
                       ),
                       (a, b) -> a
                   );
    }

    private LayerMap<Deltas> deltas(
        LayerMap<Outputs> outputs,
        LossFunction lossFunction,
        Labels labels
    ) {
        return networkLayers.backwardPropagationSegments(networkWeights)
                   .stream()
                   .reduce(
                       new LayerMap<>(
                           networkLayers.outputLayer().index(),
                           networkLayers.outputLayer().deltas(outputs, lossFunction, labels)
                       ),
                       (deltas, segment) -> deltas.with(
                           segment.participant().index(),
                           segment.deltas(deltas, outputs)
                       ),
                       (a, b) -> a
                   );
    }
}
