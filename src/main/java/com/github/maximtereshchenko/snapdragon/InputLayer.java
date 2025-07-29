package com.github.maximtereshchenko.snapdragon;

record InputLayer(int size) implements Layer {

    @Override
    public LayerIndex index() {
        return new LayerIndex(0);
    }

    Outputs outputs(Inputs inputs) {
        var tensor = inputs.tensor();
        var shape = tensor.shape();
        if (shape.length != 2 && shape[shape.length - 1] != size) {
            throw new IllegalArgumentException();
        }
        return new Outputs(tensor);
    }
}
