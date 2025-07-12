package com.github.maximtereshchenko.snapdragon;

record InputLayer(int size) implements Layer {

    @Override
    public LayerIndex index() {
        return new LayerIndex(0);
    }

    Outputs outputs(Inputs inputs) {
        var matrix = inputs.matrix();
        if (matrix.columns() != size) {
            throw new IllegalArgumentException();
        }
        return new Outputs(matrix);
    }
}
