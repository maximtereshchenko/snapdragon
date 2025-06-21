package com.github.maximtereshchenko.snapdragon;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class NeuralNetwork {

    private final List<Matrix> weights;
    private final List<Matrix> biases;
    private final ActivationFunction hiddenLayerActivationFunction;
    private final ActivationFunction outputLayerActivationFunction;

    private NeuralNetwork(
        List<Matrix> weights,
        List<Matrix> biases,
        ActivationFunction hiddenLayerActivationFunction,
        ActivationFunction outputLayerActivationFunction
    ) {
        this.weights = weights;
        this.biases = biases;
        this.hiddenLayerActivationFunction = hiddenLayerActivationFunction;
        this.outputLayerActivationFunction = outputLayerActivationFunction;
    }

    public static NeuralNetwork from(
        List<Matrix> weights,
        List<Matrix> biases,
        ActivationFunction hiddenLayerActivationFunction,
        ActivationFunction outputLayerActivationFunction
    ) {
        if (weights.isEmpty() || weights.size() != biases.size()) {
            throw new IllegalArgumentException();
        }
        for (var i = 0; i < weights.size(); i++) {
            var weightMatrix = weights.get(i);
            if (
                weightMatrix.columns() != biases.get(i).columns() ||
                    i > 0 && weightMatrix.rows() != biases.get(i - 1).columns()
            ) {
                throw new IllegalArgumentException();
            }
        }
        return new NeuralNetwork(
            weights,
            biases,
            hiddenLayerActivationFunction,
            outputLayerActivationFunction
        );
    }

    public Matrix prediction(Matrix inputs) {
        return outputs(inputs).getLast();
    }

    public NeuralNetwork adjusted(
        Matrix inputs,
        Matrix labels,
        LossFunction lossFunction,
        double learningRate
    ) {
        var outputs = outputs(inputs);
        var deltas = new Matrix[biases.size()];
        deltas[deltas.length - 1] = lossFunction.derivative(outputs.getLast(), labels)
                                        .combined(
                                            outputLayerActivationFunction.derivative(outputs.getLast()),
                                            (a, b) -> a * b
                                        );
        for (var i = biases.size() - 2; i >= 0; i--) {
            deltas[i] = deltas[i + 1].product(
                    weights.get(i + 1)
                        .transposed()
                )
                            .combined(
                                hiddenLayerActivationFunction.derivative(outputs.get(i + 1)),
                                (a, b) -> a * b
                            );
        }
        var adjustedWeights = new ArrayList<Matrix>();
        for (var i = 0; i < weights.size(); i++) {
            adjustedWeights.add(
                weights.get(i)
                    .combined(
                        outputs.get(i)
                            .transposed()
                            .product(deltas[i])
                            .applied(value -> value / inputs.rows())
                            .applied(value -> learningRate * value),
                        (a, b) -> a - b
                    )
            );
        }
        var adjustedBiases = new ArrayList<Matrix>();
        for (var i = 0; i < biases.size(); i++) {
            var delta = deltas[i];
            var vector = new double[delta.columns()];
            for (var row = 0; row < delta.rows(); row++) {
                for (var column = 0; column < delta.columns(); column++) {
                    vector[column] += delta.value(row, column);
                }
            }
            adjustedBiases.add(
                biases.get(i)
                    .combined(
                        Matrix.horizontalVector(vector)
                            .applied(value -> learningRate * value / inputs.rows()),
                        (a, b) -> a - b
                    )
            );
        }
        return NeuralNetwork.from(
            adjustedWeights,
            adjustedBiases,
            hiddenLayerActivationFunction,
            outputLayerActivationFunction
        );
    }

    @Override
    public int hashCode() {
        return Objects.hash(weights, biases, hiddenLayerActivationFunction, outputLayerActivationFunction);
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        return object instanceof NeuralNetwork that &&
                   Objects.equals(weights, that.weights) &&
                   Objects.equals(biases, that.biases) &&
                   Objects.equals(hiddenLayerActivationFunction, that.hiddenLayerActivationFunction) &&
                   Objects.equals(outputLayerActivationFunction, that.outputLayerActivationFunction);
    }

    @Override
    public String toString() {
        return "NeuralNetwork{" +
                   "weights=" + weights +
                   ", biases=" + biases +
                   ", hiddenLayerActivationFunction=" + hiddenLayerActivationFunction +
                   ", outputLayerActivationFunction=" + outputLayerActivationFunction +
                   '}';
    }

    private List<Matrix> outputs(Matrix inputs) {
        if (weights.getFirst().rows() != inputs.columns()) {
            throw new IllegalArgumentException();
        }
        var outputs = new ArrayList<Matrix>();
        outputs.add(inputs);
        for (var i = 0; i < weights.size(); i++) {
            var weightedSums = outputs.getLast().product(weights.get(i));
            outputs.add(
                activationFunction(i)
                    .apply(
                        weightedSums.combined(
                            biases.get(i)
                                .broadcasted(
                                    weightedSums.rows(),
                                    weightedSums.columns()
                                ),
                            Double::sum
                        )
                    )
            );
        }
        return outputs;
    }

    private ActivationFunction activationFunction(int index) {
        if (index == weights.size() - 1) {
            return outputLayerActivationFunction;
        }
        return hiddenLayerActivationFunction;
    }
}
