package com.github.maximtereshchenko.snapdragon;

final class BackwardPropagationSegment implements Segment<HiddenLayer> {

    private final HiddenLayer left;
    private final Layer right;
    private final Weights weights;

    BackwardPropagationSegment(HiddenLayer left, Layer right, Weights weights) {
        this.left = left;
        this.right = right;
        this.weights = weights;
    }

    @Override
    public HiddenLayer participant() {
        return left;
    }

    Deltas deltas(LayerMap<Deltas> deltas, LayerMap<Outputs> outputs) {
        return left.deltas(
            deltas.element(right.index()),
            outputs.element(left.index()),
            weights
        );
    }
}
