package com.github.maximtereshchenko.snapdragon.neuronnetwork;

import com.github.maximtereshchenko.snapdragon.matrix.Matrix;

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
        var lastOutputs = outputs.getLast();
        var lastDeltas = lossFunction.derivative(lastOutputs, labels)
                             .combined(
                                 outputLayerActivationFunction.derivative(lastOutputs),
                                 (a, b) -> a * b
                             );
        var firstOutputs = outputs.getFirst();
        var weightGradient = firstOutputs.transposed()
                                 .product(lastDeltas)
                                 .applied(value -> value / inputs.rows());
        var biasGradient = lastDeltas.applied(value -> value / inputs.rows());
        return NeuralNetwork.from(
            List.of(
                weights.getFirst()
                    .combined(
                        weightGradient.applied(value -> value * learningRate),
                        (a, b) -> a - b
                    )
            ),
            List.of(
                biases.getFirst()
                    .combined(
                        biasGradient.applied(value -> value * learningRate),
                        (a, b) -> a - b
                    )
            ),
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
