package com.github.maximtereshchenko.snapdragon;

final class ForwardPropagationSegment implements Segment<ForwardPropagationParticipant<?>> {

    private final Layer left;
    private final ForwardPropagationParticipant<?> right;
    private final Weights weights;

    ForwardPropagationSegment(Layer left, ForwardPropagationParticipant<?> right, Weights weights) {
        this.left = left;
        this.right = right;
        this.weights = weights;
    }

    @Override
    public ForwardPropagationParticipant<?> participant() {
        return right;
    }

    Outputs outputs(LayerMap<Outputs> networkOutputs) {
        return right.outputs(networkOutputs.element(left.index()), weights);
    }
}
