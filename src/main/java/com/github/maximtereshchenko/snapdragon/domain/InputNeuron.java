package com.github.maximtereshchenko.snapdragon.domain;

final class InputNeuron implements Neuron {

    private final int index;

    InputNeuron(int index) {
        this.index = index;
    }

    @Override
    public double value(double[] inputs) {
        return inputs[index];
    }
}
