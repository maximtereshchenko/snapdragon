package com.github.maximtereshchenko.snapdragon;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

final class NetworkLayers {

    private final InputLayer inputLayer;
    private final List<HiddenLayer> hiddenLayers;
    private final OutputLayer outputLayer;

    NetworkLayers(InputLayer inputLayer, List<HiddenLayer> hiddenLayers, OutputLayer outputLayer) {
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
    }

    @Override
    public int hashCode() {
        return Objects.hash(inputLayer, hiddenLayers, outputLayer);
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        return object instanceof NetworkLayers that &&
                   Objects.equals(inputLayer, that.inputLayer) &&
                   Objects.equals(hiddenLayers, that.hiddenLayers) &&
                   Objects.equals(outputLayer, that.outputLayer);
    }

    @Override
    public String toString() {
        return "NetworkLayers{" +
                   "inputLayer=" + inputLayer +
                   ", hiddenLayers=" + hiddenLayers +
                   ", outputLayer=" + outputLayer +
                   '}';
    }

    InputLayer inputLayer() {
        return inputLayer;
    }

    OutputLayer outputLayer() {
        return outputLayer;
    }

    List<ForwardPropagationSegment> forwardPropagationSegments(NetworkWeights networkWeights) {
        var segments = new ArrayList<ForwardPropagationSegment>();
        Layer left = inputLayer;
        for (var hiddenLayer : hiddenLayers) {
            segments.add(forwardPropagationSegment(left, hiddenLayer, networkWeights));
            left = hiddenLayer;
        }
        segments.add(forwardPropagationSegment(left, outputLayer, networkWeights));
        return segments;
    }

    List<BackwardPropagationSegment> backwardPropagationSegments(NetworkWeights networkWeights) {
        var segments = new ArrayList<BackwardPropagationSegment>();
        Layer right = outputLayer;
        for (var hiddenLayer : hiddenLayers.reversed()) {
            segments.add(
                new BackwardPropagationSegment(
                    hiddenLayer,
                    right,
                    networkWeights.weights(hiddenLayer.index(), right.index())
                )
            );
            right = hiddenLayer;
        }
        return segments;
    }

    NetworkLayers calibrated(LayerMap<Deltas> deltas, LearningRate learningRate) {
        return new NetworkLayers(
            inputLayer,
            hiddenLayers.stream()
                .map(hiddenLayer ->
                         hiddenLayer.calibrated(deltas.element(hiddenLayer.index()), learningRate)
                )
                .toList(),
            outputLayer.calibrated(deltas.element(outputLayer.index()), learningRate)
        );
    }

    private ForwardPropagationSegment forwardPropagationSegment(
        Layer left,
        ForwardPropagationParticipant<?> right,
        NetworkWeights networkWeights
    ) {
        return new ForwardPropagationSegment(
            left,
            right,
            networkWeights.weights(left.index(), right.index())
        );
    }
}
