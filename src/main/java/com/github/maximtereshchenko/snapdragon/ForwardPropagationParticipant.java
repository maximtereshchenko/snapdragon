package com.github.maximtereshchenko.snapdragon;

import java.util.Objects;

abstract class ForwardPropagationParticipant<T extends ForwardPropagationParticipant<T>>
    implements Layer {

    private final LayerIndex index;
    private final Biases biases;
    private final ActivationFunction activationFunction;

    ForwardPropagationParticipant(
        LayerIndex index,
        Biases biases,
        ActivationFunction activationFunction
    ) {
        this.index = index;
        this.biases = biases;
        this.activationFunction = activationFunction;
    }

    @Override
    public LayerIndex index() {
        return index;
    }

    @Override
    public int size() {
        return biases.size();
    }

    @Override
    public int hashCode() {
        return Objects.hash(index, biases, activationFunction);
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        return object instanceof ForwardPropagationParticipant<?> that &&
                   Objects.equals(index, that.index) &&
                   Objects.equals(biases, that.biases) &&
                   Objects.equals(activationFunction, that.activationFunction);
    }

    @Override
    public String toString() {
        return "ForwardPropagationParticipant{" +
                   "index=" + index +
                   ", biases=" + biases +
                   ", activationFunction=" + activationFunction +
                   '}';
    }

    ActivationFunction activationFunction() {
        return activationFunction;
    }

    final Outputs outputs(Outputs outputs, Weights weights) {
        var weightedSums = outputs.matrix().product(weights.matrix());
        return new Outputs(
            activationFunction.apply(
                weightedSums.combined(
                    biases.matrix()
                        .broadcasted(
                            weightedSums.rows(),
                            weightedSums.columns()
                        ),
                    Double::sum
                )
            )
        );
    }

    final T calibrated(Deltas deltas, LearningRate learningRate) {
        return calibrated(
            index,
            biases.calibrated(deltas, learningRate),
            activationFunction
        );
    }

    abstract T calibrated(
        LayerIndex index,
        Biases calibrated,
        ActivationFunction activationFunction
    );
}
