package com.github.maximtereshchenko.snapdragon.domain;

final class PredictingInputNeuron implements PredictingNeuron {

    private final int index;

    PredictingInputNeuron(int index) {
        this.index = index;
    }

    @Override
    public double prediction(double[] inputs) {
        return inputs[index];
    }
}
