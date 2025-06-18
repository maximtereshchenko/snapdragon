package com.github.maximtereshchenko.snapdragon.neuronnetwork.domain;

public final class Sigmoid implements ActivationFunction {

    @Override
    public double apply(double weightedSum) {
        return 1 / (1 + Math.pow(Math.E, -weightedSum));
    }
}
